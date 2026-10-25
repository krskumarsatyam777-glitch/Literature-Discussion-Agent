from graph import checkpointer,pool


def init_threads_table():
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


def upsert_thread(thread_id: str, title: str):
    """Create the thread on its first turn; afterwards only bump updated_at."""
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO threads (id, title) VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE SET updated_at = now()
            """,
            (thread_id, title),
        )


def list_threads(limit: int = 30):
    with pool.connection() as conn:
        return conn.execute(
            "SELECT id, title FROM threads ORDER BY updated_at DESC LIMIT %s",
            (limit,),
        ).fetchall()

def delete_thread(thread_id: str):
    """Remove a chat completely: its saved conversation and its sidebar entry."""
    checkpointer.delete_thread(thread_id)
    with pool.connection() as conn:
        conn.execute("DELETE FROM threads WHERE id = %s", (thread_id,))