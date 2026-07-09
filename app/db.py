"""app/db.py — Postgres connection + migration runner.

Phase 1 is dense-only: this module owns the `chunks` table and pgvector
registration. Lexical/FTS wiring arrives in Phase 5's own migration.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app.config import Settings, get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"


def get_connection(settings: Settings | None = None) -> psycopg.Connection:
    """Open a connection with pgvector types registered and autocommit on.

    Autocommit suits this app's usage: single statements or migration scripts,
    no multi-statement transactions to coordinate yet.
    """
    settings = settings or get_settings()
    conn = psycopg.connect(settings.database_url, autocommit=True)
    # The vector type must exist before psycopg can look up its OID.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def apply_migrations(conn: psycopg.Connection) -> list[str]:
    """Run any .sql files in db/migrations not yet recorded as applied.

    Returns the filenames applied this call (empty if already up to date).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    applied = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations").fetchall()}

    newly_applied = []
    for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if migration_path.name in applied:
            continue
        conn.execute(migration_path.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO schema_migrations (filename) VALUES (%s)",
            (migration_path.name,),
        )
        newly_applied.append(migration_path.name)

    return newly_applied
