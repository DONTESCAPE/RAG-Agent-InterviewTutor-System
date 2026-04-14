from typing import List
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from agent.agent_tools import rag_summarize
from model.factory import chat_model
from rag.rag_service import RagSummarizeService
from utils.prompt_loader import load_report_prompts, load_system_prompts, load_system_prompts2


class InterviewAssistantService:
    def __init__(self):
        self.rag_service = RagSummarizeService()
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

    def interview_chat(self, user_input: str, history: List[dict]) -> str:
        messages = self._to_agent_messages(history)
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_input:
            messages.append({"role": "user", "content": user_input})
        response = self.interview_executor.invoke(
            {"messages": messages}
        )
        output = self._extract_ai_output(response)
        if output:
            return output
        # 兜底：避免agent返回结构变化导致空回复
        return self.rag_service.rag_summarize(user_input)

    def qa_chat(self, user_input: str, history: List[dict]) -> str:
        messages = self._to_agent_messages(history)
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_input:
            messages.append({"role": "user", "content": user_input})
        response = self.qa_executor.invoke(
            {"messages": messages}
        )
        output = self._extract_ai_output(response)
        if output:
            return output
        # 兜底：保证问答模式至少返回RAG总结结果
        return self.rag_service.rag_summarize(user_input)

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

    def generate_report(self, interview_history: List[dict], interview_questions: List[str]) -> str:
        full_log = []
        for message in interview_history:
            role = "候选人" if message["role"] == "user" else "面试官"
            full_log.append(f"{role}：{message['content']}")

        questions_text = "\n".join([f"{idx + 1}. {question}" for idx, question in enumerate(interview_questions)])
        question_query = "；".join(interview_questions) if interview_questions else "本次面试问题"

        docs = self.rag_service.retriever_docs(question_query)
        references = []
        for idx, doc in enumerate(docs, start=1):
            references.append(f"【参考资料{idx}】{doc.page_content}")

        interview_log = (
            f"【本次面试问题】\n{questions_text}\n\n"
            f"【完整对话记录】\n{chr(10).join(full_log)}\n\n"
            f"【知识库参考】\n{chr(10).join(references)}"
        )

        return self.report_chain.invoke({"interview_log": interview_log})
