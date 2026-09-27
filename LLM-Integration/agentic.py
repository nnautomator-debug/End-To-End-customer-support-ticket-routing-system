import json
import os

import duckdb
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "feedback_tickets.duckdb")
TABLE = "ticket_disagreements"

AGENT_MODEL = "openai/gpt-oss-20b"

COLUMNS = [
    ("body", "TEXT"),
    ("routing_queue", "TEXT"),
    ("primary_topic", "TEXT"),
    ("topic_subcategory", "TEXT"),
    ("topic_aspect", "TEXT"),
    ("supplementary_topic", "TEXT"),
    ("target", "TEXT"),
]


def _connect():
    return duckdb.connect(DB_PATH)


def init_db():
    cols = ", ".join(f"{name} {typ}" for name, typ in COLUMNS)
    with _connect() as con:
        con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ({cols})")


def insert_ticket(data: dict) -> str:
    cols = [name for name, _ in COLUMNS]
    values = [str(data.get(name, "")) for name in cols]
    placeholders = ", ".join("?" for _ in cols)
    with _connect() as con:
        rows = con.execute(
            f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"RETURNING target",
            values,
        ).fetchall()
    return str(rows[0][0])


def count_disagreements() -> int:
    with _connect() as con:
        return int(con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])


@tool
def append_disagreement(ticket_json: str) -> str:
    """Persist a ticket where the RF model and the LLM judge disagreed.

    Args:
        ticket_json: JSON string with keys body, routing_queue, primary_topic,
            topic_subcategory, topic_aspect, supplementary_topic, and target
            (the LLM-corrected priority).
    """
    data = json.loads(ticket_json)
    inserted = insert_ticket(data)
    return (f"Stored ticket — queue={data.get('routing_queue')}, "
            f"target={inserted}, total rows={count_disagreements()}.")


def build_agent():
    llm = ChatGroq(model=AGENT_MODEL, temperature=0)
    prompt = (
        "You are the data-collection agent in a self-learning ticket priority pipeline.\n"
        "You receive a JSON payload describing a ticket where the ML model and the LLM "
        "judge disagreed. Persist it to the feedback database by calling the "
        "append_disagreement tool with the exact JSON payload.\n"
        "Then reply in one short sentence with what you stored."
    )
    return create_react_agent(llm, [append_disagreement], prompt=prompt)


def record_disagreement(ticket: dict, agent=None) -> str:
    if agent is None:
        agent = build_agent()
    result = agent.invoke(
        {"messages": [("user", json.dumps(ticket, ensure_ascii=False))]}
    )
    return result["messages"][-1].content


init_db()