"""Database engine and session helpers for PostgreSQL/PostGIS."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

DEFAULT_DATABASE_URL = "postgresql+psycopg://forest_user:forest_pass@localhost:5432/forest_monitoring"
SCHEMA_SQL_PATH = Path(__file__).resolve().parents[3] / "database" / "schema.sql"


def get_database_url(explicit_url: str | None = None) -> str:
    return explicit_url or os.getenv("FOREST_MONITORING_DATABASE_URL") or DEFAULT_DATABASE_URL


def create_db_engine(database_url: str | None = None, *, echo: bool = False) -> Engine:
    return create_engine(get_database_url(database_url), echo=echo, future=True, pool_pre_ping=True)


def create_session_factory(database_url: str | None = None, *, echo: bool = False) -> sessionmaker[Session]:
    engine = create_db_engine(database_url, echo=echo)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


@contextmanager
def session_scope(database_url: str | None = None, *, echo: bool = False) -> Iterator[Session]:
    factory = create_session_factory(database_url, echo=echo)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all_tables(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def ensure_postgis_extension(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))


def initialize_database(engine: Engine, schema_path: Path | None = None) -> None:
    schema_file = schema_path or SCHEMA_SQL_PATH
    sql_text = schema_file.read_text()
    statements = [statement.strip() for statement in sql_text.split(";") if statement.strip()]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
