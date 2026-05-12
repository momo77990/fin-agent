"""用户级 API 配置（LLM provider / base_url / api_key / model / tavily_key）。"""

from typing import Optional

from sqlalchemy import select

from db.models import UserSettings
from db.session import session_scope


_ALLOWED_FIELDS = {
    "llm_provider",
    "llm_base_url",
    "llm_api_key",
    "llm_model",
    "embedding_model",
    "tavily_api_key",
}


def get_settings(user_id: int) -> Optional[dict]:
    """返回当前用户的设置（含 api_key 明文）；尚未配置则返回 None。"""
    with session_scope() as session:
        s = session.scalar(select(UserSettings).where(UserSettings.user_id == user_id))
        if s is None:
            return None
        return {
            "llm_provider": s.llm_provider,
            "llm_base_url": s.llm_base_url,
            "llm_api_key": s.llm_api_key,
            "llm_model": s.llm_model,
            "embedding_model": s.embedding_model,
            "tavily_api_key": s.tavily_api_key,
        }


def upsert_settings(user_id: int, **fields) -> None:
    """新增或更新当前用户的设置；空字符串归一化为 None。"""
    cleaned = {}
    for k, v in fields.items():
        if k not in _ALLOWED_FIELDS:
            continue
        if isinstance(v, str):
            v = v.strip() or None
        cleaned[k] = v

    with session_scope() as session:
        s = session.scalar(select(UserSettings).where(UserSettings.user_id == user_id))
        if s is None:
            s = UserSettings(user_id=user_id, **cleaned)
            session.add(s)
        else:
            for k, v in cleaned.items():
                setattr(s, k, v)


def is_llm_configured(settings: Optional[dict]) -> bool:
    if not settings:
        return False
    return bool(settings.get("llm_api_key") and settings.get("llm_model"))
