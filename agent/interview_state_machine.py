from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from agent.interview_policy import InterviewDecision


@dataclass
class InterviewStateDecision:
    action: str
    reply_mode: str
    should_record_question: bool = False
    should_record_hint: bool = False
    should_end: bool = False
    reason: str = ""


class InterviewStateMachine:
    """面试状态机：管理岗位、题目轮次、追问与切题。"""

    def __init__(self):
        self.max_followups_per_question = 2
        self.max_same_question_turns = 3

    @staticmethod
    def _get_state_value(state: Dict, key: str, default=None):
        return state.get(key, default) if isinstance(state, dict) else default

    def ensure_state(self, state: Dict | None = None) -> Dict:
        state = dict(state or {})
        state.setdefault("target_role", "")
        state.setdefault("current_question", "")
        state.setdefault("current_question_index", 0)
        state.setdefault("followup_count", 0)
        state.setdefault("turn_count", 0)
        state.setdefault("awaiting_answer", False)
        state.setdefault("finished", False)
        state.setdefault("role_selected", bool(state.get("target_role")))
        return state

    def start_interview(self, role: str) -> Dict:
        state = self.ensure_state({})
        state["target_role"] = (role or "").strip()
        state["current_question"] = ""
        state["current_question_index"] = 0
        state["followup_count"] = 0
        state["turn_count"] = 0
        state["awaiting_answer"] = False
        state["finished"] = False
        state["role_selected"] = bool(state["target_role"])
        return state

    def update_current_question(self, state: Dict, question: str, is_followup: bool = False) -> Dict:
        state = self.ensure_state(state)
        state["current_question"] = question or ""
        state["awaiting_answer"] = True
        state["finished"] = False
        if not is_followup:
            state["followup_count"] = 0
            state["current_question_index"] = int(state.get("current_question_index", 0)) + 1
        return state

    def mark_answered(self, state: Dict) -> Dict:
        state = self.ensure_state(state)
        state["turn_count"] = int(state.get("turn_count", 0)) + 1
        state["awaiting_answer"] = False
        return state

    def decide_next_action(
        self,
        decision: InterviewDecision,
        current_state: Dict,
        interview_history: Sequence[dict],
        interview_questions: Sequence[str],
    ) -> InterviewStateDecision:
        state = self.ensure_state(current_state)
        if state.get("finished"):
            return InterviewStateDecision(action="finish", reply_mode="finish", should_end=True, reason="面试已结束")

        if decision.should_end:
            state["finished"] = True
            return InterviewStateDecision(action="finish", reply_mode="finish", should_end=True, reason=decision.reason or "用户结束面试")

        if decision.intent in {"ask_hint", "chat_interrupt", "out_of_scope"} or decision.should_give_hint:
            return InterviewStateDecision(action="hint", reply_mode="hint", should_record_hint=True, reason=decision.reason or "需要提示")

        current_followups = int(state.get("followup_count", 0))
        if current_followups >= self.max_followups_per_question:
            state["followup_count"] = 0
            return InterviewStateDecision(action="next_question", reply_mode="next_question", should_record_question=True, reason="达到当前题目追问上限")

        if len(interview_questions) == 0:
            return InterviewStateDecision(action="first_question", reply_mode="next_question", should_record_question=True, reason="初始化第一题")

        if decision.confidence >= 0.75:
            if current_followups < self.max_followups_per_question:
                state["followup_count"] = current_followups + 1
                return InterviewStateDecision(action="follow_up", reply_mode="follow_up", should_record_question=True, reason="回答较好，继续深挖")
            return InterviewStateDecision(action="next_question", reply_mode="next_question", should_record_question=True, reason="回答已足够，切换下一题")

        if decision.confidence >= 0.45:
            state["followup_count"] = current_followups + 1
            return InterviewStateDecision(action="follow_up", reply_mode="follow_up", should_record_question=True, reason="回答一般，适当追问")

        return InterviewStateDecision(action="hint", reply_mode="hint", should_record_hint=True, reason="回答质量偏低")

    def should_ask_next_question(self, state: Dict, interview_questions: Sequence[str]) -> bool:
        state = self.ensure_state(state)
        if state.get("finished"):
            return False
        if not state.get("target_role"):
            return False
        return bool(interview_questions) or state.get("current_question_index", 0) > 0
