"""用户注册与认证。"""

import re
from typing import Optional

import bcrypt
from sqlalchemy import select

from db.models import User
from db.session import session_scope


_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,32}$")


class AuthError(Exception):
    """业务可见的认证错误。"""


def _validate_username(username: str) -> None:
    if not _USERNAME_PATTERN.fullmatch(username or ""):
        raise AuthError("用户名需为 3-32 位字母、数字或下划线。")


def _validate_password(password: str) -> None:
    if not password or len(password) < 6 or len(password) > 64:
        raise AuthError("密码长度需在 6-64 位之间。")


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def register_user(username: str, password: str) -> dict:
    """注册新用户；返回 {id, username}。"""
    username = (username or "").strip()
    _validate_username(username)
    _validate_password(password)

    with session_scope() as session:
        exists = session.scalar(select(User).where(User.username == username))
        if exists is not None:
            raise AuthError("该用户名已被占用。")

        user = User(username=username, password_hash=hash_password(password))
        session.add(user)
        session.flush()
        return {"id": user.id, "username": user.username}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """校验凭据，成功返回 {id, username}，否则 None。"""
    username = (username or "").strip()
    if not username or not password:
        return None

    with session_scope() as session:
        user = session.scalar(select(User).where(User.username == username))
        if user is None:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return {"id": user.id, "username": user.username}
