from typing import List
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from agent.agent_tools import rag_summarize
from agent.interview_policy import InterviewPolicy
from agent.interview_role_manager import InterviewRoleManager
from agent.interview_state_machine import InterviewStateMachine
from model.factory import chat_model
from rag.rag_service import RagSummarizeService
from utils.prompt_loader import load_report_prompts, load_system_prompts, load_system_prompts2


class InterviewAssistantService:
    def __init__(self):
        self.rag_service = RagSummarizeService()
        self.policy = InterviewPolicy()
        self.state_machine = InterviewStateMachine()
        self.role_manager = InterviewRoleManager()
        self.tools = [rag_summarize]
        self.interview_executor = self._build_agent_executor(load_system_prompts())
        self.qa_executor = self._build_agent_executor(load_system_prompts2())
        self.report_chain = self._build_report_chain()

    @staticmethod
    def _to_agent_messages(history: List[dict]) -> List[dict]:
        lc_messages: List[dict] = []
        for message in history:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                lc_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                lc_messages.append({"role": "assistant", "content": content})
        return lc_messages

    def _build_agent_executor(self, system_prompt: str):
        return create_agent(
            model=chat_model,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=False,
        )

    @staticmethod
    def _build_report_chain():
        report_prompt = PromptTemplate.from_template(load_report_prompts())
        return report_prompt | chat_model | StrOutputParser()

    def start_role_interview(self, role: str, history: List[dict] | None = None) -> dict:
        state = self.state_machine.start_interview(role)
        first_question = self.role_manager.get_first_question(role)
        state = self.state_machine.update_current_question(state, first_question, is_followup=False)
        return {
            "state": state,
            "question": first_question,
        }

    def interview_chat(self, user_input: str, history: List[dict], interview_state: dict | None = None) -> dict:
        state = self.state_machine.ensure_state(interview_state)
        current_question = state.get("current_question") or self._get_current_question(history)
        decision = self.policy.classify_intent(user_input, current_question, history)
        state = self.state_machine.mark_answered(state)
        action = self.state_machine.decide_next_action(decision, state, history, [])

        if action.should_end:
            state["finished"] = True
            return {"reply": "好的，本次模拟面试先到这里。", "state": state, "action": action.action}

        if action.action == "hint":
            hint = self.policy.build_hint(user_input, current_question, history)
            state["awaiting_answer"] = True
            return {"reply": hint, "state": state, "action": action.action}

        if action.action in {"first_question", "next_question"}:
            next_index = int(state.get("current_question_index", 0))
            next_question = self.role_manager.get_next_question(state.get("target_role", ""), next_index)
            state = self.state_machine.update_current_question(state, next_question, is_followup=False)
            reply = next_question
            if action.action == "first_question":
                reply = f"你好，我们先从这个岗位开始。{next_question}"
            return {"reply": reply, "state": state, "action": action.action}

        if action.action == "follow_up":
            role = state.get("target_role", "")
            keywords = self.role_manager.get_role_keywords(role)
            follow_up = self._build_follow_up_question(current_question, keywords, history)
            state = self.state_machine.update_current_question(state, follow_up, is_followup=True)
            return {"reply": follow_up, "state": state, "action": action.action}

        messages = self._to_agent_messages(history)
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_input:
            messages.append({"role": "user", "content": user_input})
        response = self.interview_executor.invoke({"messages": messages})
        output = self._extract_ai_output(response)
        if output:
            return {"reply": output, "state": state, "action": "agent"}
        return {"reply": self.rag_service.rag_summarize(user_input, history), "state": state, "action": "rag"}

    def qa_chat(self, user_input: str, history: List[dict]) -> str:
        messages = self._to_agent_messages(history)
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_input:
            messages.append({"role": "user", "content": user_input})
        response = self.qa_executor.invoke({"messages": messages})
        output = self._extract_ai_output(response)
        if output:
            return output
        return self.rag_service.rag_summarize(user_input, history)

    @staticmethod
    def _yield_text_stream(text: str, chunk_size: int = 8):
        content = str(text or "")
        if not content:
            return
        for idx in range(0, len(content), chunk_size):
            yield content[idx : idx + chunk_size]

    def qa_chat_stream(self, user_input: str, history: List[dict]):
        messages = self._to_agent_messages(history)
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_input:
            messages.append({"role": "user", "content": user_input})
        response = self.qa_executor.invoke({"messages": messages})
        output = self._extract_ai_output(response)
        if output:
            yield from self._yield_text_stream(output)
            return
        yield from self.rag_service.rag_summarize_stream(user_input, history)

    def interview_chat_stream(self, user_input: str, history: List[dict], interview_state: dict | None = None):
        result = self.interview_chat(user_input, history, interview_state)
        reply = result.get("reply", "")
        if reply:
            yield from self._yield_text_stream(reply)

    @staticmethod
    def _extract_ai_output(response: dict) -> str:
        direct_output = response.get("output")
        if isinstance(direct_output, str) and direct_output.strip():
            return direct_output.strip()

        messages = response.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return InterviewAssistantService._message_content_to_text(message.content)
            if isinstance(message, dict):
                role = message.get("role", "")
                if role == "assistant":
                    content = message.get("content", "")
                    return InterviewAssistantService._message_content_to_text(content)
        return ""

    @staticmethod
    def _message_content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            text = content.get("text", "")
            if isinstance(text, str):
                return text
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join([part for part in text_parts if part])
        return ""

    @staticmethod
    def _get_current_question(history: List[dict]) -> str:
        for message in reversed(history):
            if message.get("role") == "assistant":
                content = str(message.get("content", "")).strip()
                if content:
                    return content
        return "请先开始本次模拟面试。"

    @staticmethod
    def _build_follow_up_question(current_question: str, keywords: List[str], history: List[dict]) -> str:
        focus = keywords[0] if keywords else "关键实现"
        return f"好的，我继续追问一下：在刚才这个问题里，你是如何处理{focus}的？"

    def generate_report(self, interview_history: List[dict], interview_questions: List[str]) -> str:
        full_log = []
        for message in interview_history:
            role = "候选人" if message["role"] == "user" else "面试官"
            full_log.append(f"{role}：{message['content']}")

        questions_text = "\n".join([f"{idx + 1}. {question}" for idx, question in enumerate(interview_questions)])
        question_query = "；".join(interview_questions) if interview_questions else "本次面试问题"

        docs = self.rag_service.retriever_docs(question_query, interview_history)
        references = []
        for idx, doc in enumerate(docs, start=1):
            references.append(f"【参考资料{idx}】{doc.page_content}")

        interview_log = (
            f"【本次面试问题】\n{questions_text}\n\n"
            f"【完整对话记录】\n{chr(10).join(full_log)}\n\n"
            f"【知识库参考】\n{chr(10).join(references)}"
        )

        return self.report_chain.invoke({"interview_log": interview_log})
