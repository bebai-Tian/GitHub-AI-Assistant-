from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def ensure_memory_state_schema(bind) -> None:
    """Additive upgrade for existing demo databases; never rewrites history."""
    inspector = inspect(bind)
    for table_name in ("conversation_memories", "thread_memories"):
        if table_name not in inspector.get_table_names():
            continue
        if "working_state" not in {column["name"] for column in inspector.get_columns(table_name)}:
            with bind.begin() as connection:
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN working_state JSON"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
