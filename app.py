import os
import time
import uuid
import re
import streamlit as st

from agent.interview_assistant_service import InterviewAssistantService
from agent.agent_tools import get_city, get_weather
from rag.vector_store import VectorStoreService
from utils.user_history_store import load_user_state, save_user_state


st.set_page_config(page_title="基于RAG与Agent的多模态面试辅导助手", page_icon="💼", layout="wide")
st.title("💼 基于RAG与Agent的多模态面试辅导助手")


if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = f"guest_{uuid.uuid4().hex[:8]}"
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "interview_finished" not in st.session_state:
    st.session_state.interview_finished = False
if "interview_report" not in st.session_state:
    st.session_state.interview_report = ""
if "interview_state" not in st.session_state:
    st.session_state.interview_state = {}
if "pending_role_input" not in st.session_state:
    st.session_state.pending_role_input = ""
if "role_selected" not in st.session_state:
    st.session_state.role_selected = False
if "qa_stream_output" not in st.session_state:
    st.session_state.qa_stream_output = ""


def persist_state():
    save_user_state(
        st.session_state.current_user_id,
        {
            "interview_history": st.session_state.interview_history,
            "qa_history": st.session_state.qa_history,
            "interview_questions": st.session_state.interview_questions,
            "interview_started": st.session_state.interview_started,
            "interview_finished": st.session_state.interview_finished,
            "interview_report": st.session_state.interview_report,
            "interview_state": st.session_state.interview_state,
        },
    )


def render_chat_history(messages):
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def generate_life_advice(weather_text: str) -> tuple[str, str]:
    if not weather_text:
        return "穿衣建议：暂无", "出行提醒：暂无"

    clothing = "穿衣建议：常规穿搭即可。"
    travel = "出行提醒：保持出行节奏，注意补水。"

    temp_match = re.search(r"气温\s*([\-]?\d+)", weather_text)
    temp = int(temp_match.group(1)) if temp_match else None
    lower_weather = weather_text.lower()

    if temp is not None:
        if temp <= 5:
            clothing = "面试穿衣建议：天气偏冷，建议厚外套/羽绒服+衬衫+加绒西装裤，整洁又干练，让面试官眼前一新！"
        elif temp <= 15:
            clothing = "面试穿衣建议：建议风衣+衬衫+厚西装裤，保暖不臃肿，你就是面试场上最靓的崽！"
        elif temp <= 26:
            clothing = "面试穿衣建议：温度舒适，建议穿白衬衫+西装裤，更得体哦~"
        else:
            clothing = "面试穿衣建议：天气较热，建议纯色短袖衬衫+垂感/直筒西装裤，显出你的重视！"

    if any(k in lower_weather for k in ["雨", "雷", "阵雨", "暴雨"]):
        travel = "出行提醒：可能降雨，建议带伞，注意路滑和交通安全。"
    elif any(k in lower_weather for k in ["雪", "冰"]):
        travel = "出行提醒：可能有雨雪结冰，建议减速慢行，注意防滑。"
    elif any(k in lower_weather for k in ["雾", "霾"]):
        travel = "出行提醒：能见度或空气质量一般，建议佩戴口罩并减少久留户外。"
    elif any(k in lower_weather for k in ["大风", "风"]):
        travel = "出行提醒：风力较大，注意高空坠物，骑行请减速。"

    return clothing, travel


def get_sidebar_weather_info() -> tuple[str, str]:
    try:
        city = get_city.invoke({})
    except Exception:
        city = "未知城市"

    try:
        weather_text = get_weather.invoke({"city": city})
    except Exception:
        weather_text = "天气获取失败，请稍后重试。"
    return str(city), str(weather_text)


# 首次进入加载用户数据（只在当前用户未显式切换时执行）
if "user_state_loaded" not in st.session_state:
    loaded = load_user_state(st.session_state.current_user_id)
    st.session_state.interview_history = loaded.get("interview_history", [])
    st.session_state.qa_history = loaded.get("qa_history", [])
    st.session_state.interview_questions = loaded.get("interview_questions", [])
    st.session_state.interview_started = loaded.get("interview_started", False)
    st.session_state.interview_finished = loaded.get("interview_finished", False)
    st.session_state.interview_report = loaded.get("interview_report", "")
    st.session_state.interview_state = loaded.get("interview_state", {})
    st.session_state.role_selected = bool(st.session_state.interview_state.get("target_role"))
    st.session_state.user_state_loaded = True
    os.environ["CURRENT_USER_ID"] = st.session_state.current_user_id


service = InterviewAssistantService()

st.sidebar.header("用户管理")
user_id_input = st.sidebar.text_input("用户 ID", value=st.session_state.current_user_id)
if st.sidebar.button("切换/加载用户", use_container_width=True):
    target_user_id = user_id_input.strip() or "guest"
    loaded = load_user_state(target_user_id)
    st.session_state.current_user_id = target_user_id
    st.session_state.interview_history = loaded.get("interview_history", [])
    st.session_state.qa_history = loaded.get("qa_history", [])
    st.session_state.interview_questions = loaded.get("interview_questions", [])
    st.session_state.interview_started = loaded.get("interview_started", False)
    st.session_state.interview_finished = loaded.get("interview_finished", False)
    st.session_state.interview_report = loaded.get("interview_report", "")
    st.session_state.interview_state = loaded.get("interview_state", {})
    st.session_state.role_selected = bool(st.session_state.interview_state.get("target_role"))
    os.environ["CURRENT_USER_ID"] = target_user_id
    st.sidebar.success(f"已加载用户：{target_user_id}")
    st.rerun()

st.sidebar.header("知识库管理")
st.sidebar.write("首次使用建议先加载知识库。")
if st.sidebar.button("加载/更新知识库", use_container_width=True):
    with st.sidebar:
        with st.spinner("正在加载知识库..."):
            VectorStoreService().load_document()
        st.success("知识库加载完成")

st.sidebar.header("天气与出行建议")
city_name, weather_text = get_sidebar_weather_info()
dress_advice, travel_advice = generate_life_advice(weather_text)
st.sidebar.caption(f"当前城市：{city_name}")
st.sidebar.caption(f"实时天气：{weather_text}")
st.sidebar.caption(dress_advice)
st.sidebar.caption(travel_advice)


st.sidebar.header("模式切换")
mode = st.sidebar.radio("请选择模式", ["问答模式", "模拟面试"])

if mode == "问答模式":
    st.subheader("问答模式")
    st.caption("用户提问，模型结合知识库与自身知识进行回答。")

    if st.button("清空问答历史", use_container_width=False):
        st.session_state.qa_history = []
        st.session_state.qa_stream_output = ""
        persist_state()
        st.rerun()

    render_chat_history(st.session_state.qa_history)

    question = st.chat_input("请输入你想问的问题...")
    if question:
        st.session_state.qa_history.append({"role": "user", "content": question})
        st.session_state.qa_stream_output = ""
        persist_state()

        answer_placeholder = st.chat_message("assistant").empty()
        collected = []
        for chunk in service.qa_chat_stream(question, st.session_state.qa_history):
            collected.append(chunk)
            answer_placeholder.markdown("".join(collected))
        answer = "".join(collected).strip()
        if not answer:
            answer = "抱歉，我这次没有成功生成回答。请重试一次，或先点击左侧“加载/更新知识库”后再提问。"
            answer_placeholder.markdown(answer)
        st.session_state.qa_history.append({"role": "assistant", "content": answer})
        persist_state()
        st.rerun()
else:
    st.subheader("模拟面试模式")
    st.caption("先选择面试岗位，再由系统根据岗位生成更专业的面试问题。")

    role_options = ["后端开发", "前端开发", "数据分析", "算法工程", "产品经理", "通用技术岗位"]
    role_input = st.selectbox("请选择面试岗位", options=role_options, index=0 if not st.session_state.interview_state.get("target_role") else max(0, role_options.index(st.session_state.interview_state.get("target_role")) if st.session_state.interview_state.get("target_role") in role_options else 0))

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("开始/重置面试", use_container_width=True):
            selected_role = role_input.strip() or "通用技术岗位"
            start_result = service.start_role_interview(selected_role, st.session_state.interview_history)
            st.session_state.interview_history = []
            st.session_state.interview_questions = []
            st.session_state.interview_report = ""
            st.session_state.interview_started = True
            st.session_state.interview_finished = False
            st.session_state.interview_state = start_result["state"]
            st.session_state.role_selected = True
            persist_state()

            first_question = start_result["question"]
            st.session_state.interview_history.append({"role": "assistant", "content": f"你好，今天我们面试的是{selected_role}岗位。{first_question}"})
            st.session_state.interview_questions.append(first_question)
            st.session_state.interview_state = service.state_machine.update_current_question(st.session_state.interview_state, first_question, is_followup=False)
            persist_state()
            st.rerun()

    with col2:
        if st.button("结束本次面试", use_container_width=True):
            st.session_state.interview_finished = True
            if st.session_state.interview_state:
                st.session_state.interview_state["finished"] = True
            persist_state()
            st.rerun()

    with col3:
        current_role = st.session_state.interview_state.get("target_role", "未选择")
        st.write("当前岗位：", current_role)
        st.write("当前状态：", "已结束" if st.session_state.interview_finished else "进行中")

    render_chat_history(st.session_state.interview_history)

    if st.session_state.interview_started and not st.session_state.interview_finished:
        user_reply = st.chat_input("请输入你的回答...")
        if user_reply:
            st.session_state.interview_history.append({"role": "user", "content": user_reply})
            result = service.interview_chat(user_reply, st.session_state.interview_history, st.session_state.interview_state)
            st.session_state.interview_state = result.get("state", st.session_state.interview_state)
            interviewer_reply = result.get("reply", "")
            stream_placeholder = st.chat_message("assistant").empty()
            streamed_text = ""
            if interviewer_reply:
                for idx in range(0, len(interviewer_reply), 6):
                    streamed_text += interviewer_reply[idx : idx + 6]
                    stream_placeholder.markdown(streamed_text)
                    time.sleep(0.02)
            if not streamed_text.strip():
                streamed_text = "抱歉，本轮回答生成失败，请继续补充。"
                stream_placeholder.markdown(streamed_text)
            st.session_state.interview_history.append({"role": "assistant", "content": streamed_text})
            if "?" in streamed_text or "？" in streamed_text:
                st.session_state.interview_questions.append(streamed_text)
            persist_state()
            st.rerun()

    if st.session_state.interview_finished:
        want_report = st.checkbox("我希望生成本次面试报告", value=False)
        if want_report and st.button("生成面试报告", use_container_width=True):
            with st.spinner("正在生成报告..."):
                st.session_state.interview_report = service.generate_report(
                    st.session_state.interview_history,
                    st.session_state.interview_questions,
                )
                persist_state()
            st.rerun()

        if st.session_state.interview_report:
            st.markdown("### 面试报告")
            st.markdown(st.session_state.interview_report)
