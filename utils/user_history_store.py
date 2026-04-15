import json
import os
import re
from typing import Dict

from utils.path_tool import get_abs_path


USER_HISTORY_DIR = get_abs_path("data/user_histories")


def _normalize_user_id(user_id: str) -> str:
    normalized = (user_id or "").strip().lower()
    if not normalized:
        normalized = "guest"
    normalized = re.sub(r"[^a-zA-Z0-9_\-]", "_", normalized)
    return normalized


def _get_user_file(user_id: str) -> str:
    os.makedirs(USER_HISTORY_DIR, exist_ok=True)
    safe_user_id = _normalize_user_id(user_id)
    return os.path.join(USER_HISTORY_DIR, f"{safe_user_id}.json")


def load_user_state(user_id: str) -> Dict:
    file_path = _get_user_file(user_id)
    if not os.path.exists(file_path):
        return {
            "interview_history": [],
            "qa_history": [],
            "interview_questions": [],
            "interview_started": False,
            "interview_finished": False,
            "interview_report": "",
            "interview_state": {},
        }

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data.setdefault("interview_state", {})
        return data


def save_user_state(user_id: str, state: Dict) -> None:
    file_path = _get_user_file(user_id)
    payload = {
        "interview_history": state.get("interview_history", []),
        "qa_history": state.get("qa_history", []),
        "interview_questions": state.get("interview_questions", []),
        "interview_started": state.get("interview_started", False),
        "interview_finished": state.get("interview_finished", False),
        "interview_report": state.get("interview_report", ""),
        "interview_state": state.get("interview_state", {}),
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
