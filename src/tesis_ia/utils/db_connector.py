"""Módulo de conexión y utilidades para interactuar con la base de datos MySQL de la plataforma."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()


def get_db_url() -> str:
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", os.getenv("DB_PASS", ""))
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "tesis")

    from urllib.parse import quote_plus
    encoded_pass = quote_plus(password) if password else ""

    return f"mysql+pymysql://{user}:{encoded_pass}@{host}:{port}/{db_name}?charset=utf8mb4"


_engine: Engine | None = None


def get_engine() -> Engine:
    """Obtiene o crea una instancia Singleton del motor SQLAlchemy."""
    global _engine
    if _engine is None:
        url = get_db_url()
        _engine = create_engine(
            url,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_connection() -> bool:
    """Verifica si la conexión a la BD responde correctamente."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            return result == 1
    except Exception as e:
        print(f"Error conectando a la base de datos: {e}")
        return False
