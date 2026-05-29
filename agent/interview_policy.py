from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from utils.logger_handler import logger


@dataclass
class InterviewDecision:
    intent: str
    confidence: float = 0.0
    should_end: bool = False
    should_move_next: bool = False
    should_give_hint: bool = False
    reason: str = ""


class InterviewPolicy:
    """面试编排策略：意图分类 + 质量评分 + 提示生成 + 下一题决策。"""

    def __init__(self):
        self.intent_chain = self._build_intent_chain()
        self.quality_chain = self._build_quality_chain()
        self.hint_chain = self._build_hint_chain()
        self.followup_summary_chain = self._build_followup_summary_chain()
        self.max_followups = 2

    @staticmethod
    def _build_intent_chain():
        prompt = PromptTemplate.from_template(
            """你是一个技术面试对话意图分类器。请根据用户输入、当前题目和上下文，判断用户意图。

可选意图只有以下几类：
- answer_question：用户在认真回答当前面试题
- ask_hint：用户请求提示、求助或表示不会
- chat_interrupt：闲聊、打断、插话
- out_of_scope：偏离当前面试题但仍在技术讨论范围
- finish_interview：用户想结束面试

要求：
1. 只输出 JSON；
2. JSON 字段固定为 intent, confidence, reason；
3. confidence 为 0 到 1 的小数；
4. 不要输出多余解释。

### 当前题目
{current_question}

### 最近上下文
{history}

### 用户输入
{user_input}
"""
        )
        return prompt | chat_model | StrOutputParser()

    @staticmethod
    def _build_quality_chain():
        prompt = PromptTemplate.from_template(
            """你是一个技术面试回答质量评估器。请判断用户回答当前面试题的质量。

要求：
1. 只输出 JSON；
2. JSON 字段固定为 score, reason；
3. score 取 0 到 1 的小数，越高表示回答越完整、越贴题、越有技术深度；
4. 不要输出多余解释。

### 当前题目
{current_question}

### 用户回答
{user_input}

### 最近上下文
{history}
"""
        )
        return prompt | chat_model | StrOutputParser()

    @staticmethod
    def _build_hint_chain():
        prompt = PromptTemplate.from_template(
            """你是一位专业、严格但愿意引导的技术面试官。当前需要给用户一点提示，但不能直接给完整答案。

要求：
1. 只给 1-3 句简短提示；
2. 给方向，不给标准答案；
3. 尽量围绕当前题目；
4. 提示后引导用户继续回答；
5. 输出纯中文文本，不要 JSON，不要项目符号也可以。

### 当前题目
{current_question}

### 用户输入
{user_input}

### 最近上下文
{history}
"""
        )
        return prompt | chat_model | StrOutputParser()

    @staticmethod
    def _build_followup_summary_chain():
        prompt = PromptTemplate.from_template(
            """你是一位技术面试官，请从用户的回答中提取最值得追问的技术点。

要求：
1. 只输出一个最适合追问的关键词或短语；
2. 优先选择用户回答中出现的具体技术点、方案、名词或动作；
3. 如果没有明显可追问点，就返回“关键实现”；
4. 不要输出解释，不要输出多余文本。

### 当前题目
{current_question}

### 用户回答
{user_input}

### 最近上下文
{history}
"""
        )
        return prompt | chat_model | StrOutputParser()

    @staticmethod
    def _history_to_text(history: Sequence[dict], max_turns: int = 4) -> str:
        if not history:
            return "无"
        recent = history[-max_turns:]
        lines: List[str] = []
        for msg in recent:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            prefix = "用户" if role == "user" else "面试官"
            lines.append(f"{prefix}：{content}")
        return "\n".join(lines) if lines else "无"

    @staticmethod
    def _extract_json_like(text: str) -> dict:
        raw = str(text or "").strip()
        if not raw:
            return {}
        try:
            import json
            return json.loads(raw)
        except Exception:
            pass
        m_intent = re.search(r'"intent"\s*:\s*"([^"]+)"', raw)
        m_conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
        m_score = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        m_reason = re.search(r'"reason"\s*:\s*"([\s\S]*?)"', raw)
        data = {}
        if m_intent:
            data["intent"] = m_intent.group(1)
        if m_conf:
            data["confidence"] = float(m_conf.group(1))
        if m_score:
            data["score"] = float(m_score.group(1))
        if m_reason:
            data["reason"] = m_reason.group(1)
        return data

    @staticmethod
    def _simple_intent_fallback(user_input: str) -> InterviewDecision:
        text = str(user_input or "").strip().lower()
        if not text:
            return InterviewDecision(intent="ask_hint", confidence=0.0, should_give_hint=True, reason="空输入")
        if any(k in text for k in ["结束", "不面了", "先到这", "到这里", "结束吧", "stop", "finish"]):
            return InterviewDecision(intent="finish_interview", confidence=0.95, should_end=True, reason="用户结束面试")
        if any(k in text for k in ["提示", "hint", "不会", "不知道", "求助", "怎么答", "方向"]):
            return InterviewDecision(intent="ask_hint", confidence=0.9, should_give_hint=True, reason="用户请求提示")
        if any(k in text for k in ["哈哈", "谢谢", "在吗", "你好", "等等", "先别", "换个", "聊聊"]):
            return InterviewDecision(intent="chat_interrupt", confidence=0.7, should_give_hint=True, reason="闲聊/打断")
        return InterviewDecision(intent="answer_question", confidence=0.55, reason="默认视为回答")

    def classify_intent(self, user_input: str, current_question: str, history: Sequence[dict]) -> InterviewDecision:
        history_text = self._history_to_text(history)
        try:
            raw = self.intent_chain.invoke({"user_input": user_input, "current_question": current_question, "history": history_text})
            data = self._extract_json_like(raw)
            intent = str(data.get("intent", "")).strip() or "answer_question"
            confidence = float(data.get("confidence", 0.5) or 0.5)
            reason = str(data.get("reason", "")).strip()
            should_end = intent == "finish_interview"
            should_give_hint = intent in {"ask_hint", "chat_interrupt", "out_of_scope"}
            return InterviewDecision(
                intent=intent,
                confidence=confidence,
                should_end=should_end,
                should_give_hint=should_give_hint,
                reason=reason,
            )
        except Exception as e:
            logger.warning(f"[InterviewPolicy] 意图分类失败，使用规则兜底：{str(e)}")
            return self._simple_intent_fallback(user_input)

    def score_answer_quality(self, user_input: str, current_question: str, history: Sequence[dict]) -> float:
        history_text = self._history_to_text(history)
        try:
            raw = self.quality_chain.invoke({"user_input": user_input, "current_question": current_question, "history": history_text})
            data = self._extract_json_like(raw)
            score = float(data.get("score", 0.0) or 0.0)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"[InterviewPolicy] 回答质量评分失败，使用规则兜底：{str(e)}")
            text = str(user_input or "")
            if len(text.strip()) < 12:
                return 0.2
            if any(k in text for k in ["因为", "所以", "首先", "其次", "最后", "例如", "举个例子"]):
                return 0.7
            return 0.5

    def build_hint(self, user_input: str, current_question: str, history: Sequence[dict]) -> str:
        history_text = self._history_to_text(history)
        try:
            hint = self.hint_chain.invoke({"user_input": user_input, "current_question": current_question, "history": history_text})
            hint = str(hint or "").strip()
            if hint:
                return hint
        except Exception as e:
            logger.warning(f"[InterviewPolicy] 提示生成失败，回退规则提示：{str(e)}")

        return "你可以先从原理、流程、优缺点或实际场景这几个角度回答，我再继续追问。"

    def extract_followup_focus(self, user_input: str, current_question: str, history: Sequence[dict]) -> str:
        history_text = self._history_to_text(history)
        try:
            focus = self.followup_summary_chain.invoke({"user_input": user_input, "current_question": current_question, "history": history_text})
            focus = str(focus or "").strip()
            if focus:
                return focus
        except Exception as e:
            logger.warning(f"[InterviewPolicy] 追问点提取失败，使用兜底关键词：{str(e)}")
        return "关键实现"

    def should_move_next(self, history: Sequence[dict], interview_questions: Sequence[str]) -> bool:
        user_turns = sum(1 for m in history if m.get("role") == "user")
        return user_turns >= max(1, self.max_followups + 1)
