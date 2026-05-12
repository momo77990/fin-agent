"""会话与消息的 CRUD。所有操作均按 user_id 隔离。"""

import json
from typing import List, Optional

from sqlalchemy import select

from db.models import Conversation, Message
from db.session import session_scope


_DEFAULT_TITLE = "新对话"
_TITLE_MAX_LEN = 30


def _truncate_title(text: str) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= _TITLE_MAX_LEN:
        return text or _DEFAULT_TITLE
    return text[:_TITLE_MAX_LEN] + "…"


def list_conversations(user_id: int) -> List[dict]:
    with session_scope() as session:
        rows = session.scalars(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
        ).all()
        return [
            {
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
            }
            for c in rows
        ]


def create_conversation(user_id: int, title: str = _DEFAULT_TITLE) -> int:
    with session_scope() as session:
        conv = Conversation(user_id=user_id, title=title or _DEFAULT_TITLE)
        session.add(conv)
        session.flush()
        return conv.id


def rename_conversation(user_id: int, conv_id: int, new_title: str) -> bool:
    new_title = (new_title or "").strip()
    if not new_title:
        return False
    with session_scope() as session:
        conv = session.scalar(
            select(Conversation).where(
                Conversation.id == conv_id, Conversation.user_id == user_id
            )
        )
        if conv is None:
            return False
        conv.title = new_title[:200]
        return True


def delete_conversation(user_id: int, conv_id: int) -> bool:
    with session_scope() as session:
        conv = session.scalar(
            select(Conversation).where(
                Conversation.id == conv_id, Conversation.user_id == user_id
            )
        )
        if conv is None:
            return False
        session.delete(conv)
        return True


def get_messages(user_id: int, conv_id: int) -> List[dict]:
    """返回 app.py 渲染所需结构。"""
    with session_scope() as session:
        conv = session.scalar(
            select(Conversation).where(
                Conversation.id == conv_id, Conversation.user_id == user_id
            )
        )
        if conv is None:
            return []

        rows = session.scalars(
            select(Message).where(Message.conversation_id == conv_id).order_by(Message.id.asc())
        ).all()

        history: List[dict] = []
        for m in rows:
            entry = {"role": m.role, "content": m.content}
            if m.role == "assistant":
                try:
                    entry["tool_steps"] = json.loads(m.tool_steps_json) if m.tool_steps_json else []
                except json.JSONDecodeError:
                    entry["tool_steps"] = []
                entry["chart_path"] = m.chart_path
            history.append(entry)
        return history


def append_message(
    user_id: int,
    conv_id: int,
    role: str,
    content: str,
    tool_steps: Optional[list] = None,
    chart_path: Optional[str] = None,
) -> Optional[int]:
    """追加消息并刷新 updated_at；首条 user 消息且 title 仍为默认时自动重命名。"""
    if role not in ("user", "assistant"):
        raise ValueError(f"invalid role: {role}")

    with session_scope() as session:
        conv = session.scalar(
            select(Conversation).where(
                Conversation.id == conv_id, Conversation.user_id == user_id
            )
        )
        if conv is None:
            return None

        msg = Message(
            conversation_id=conv_id,
            role=role,
            content=content or "",
            tool_steps_json=json.dumps(tool_steps, ensure_ascii=False) if tool_steps else None,
            chart_path=chart_path,
        )
        session.add(msg)

        if role == "user" and conv.title == _DEFAULT_TITLE:
            conv.title = _truncate_title(content)

        session.flush()
        return msg.id
