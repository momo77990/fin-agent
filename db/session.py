"""数据库连接：engine + SessionLocal + init_db。"""

import os
from contextlib import contextmanager
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base


def _build_dsn() -> str:
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "finagent")
    user = os.getenv("DB_USER", "finagent")
    password = os.getenv("DB_PASSWORD", "finagentpass")
    return (
        f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{name}?charset=utf8mb4"
    )


engine = create_engine(
    _build_dsn(),
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    """创建所有表（幂等）。在应用启动时调用一次。"""
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope():
    """事务作用域：自动 commit/rollback/close。"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
