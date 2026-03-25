import sqlite3
import json
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

DB_FILE = "chat_history.db"


def init_db():
    """初始化 SQLite 数据库，建表"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # 创建会话表
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS sessions
                   (
                       session_id
                       TEXT
                       PRIMARY
                       KEY,
                       title
                       TEXT,
                       created_at
                       TEXT,
                       messages
                       TEXT
                   )
                   ''')
    conn.commit()
    conn.close()


def save_session(session_id, title, messages, created_at):
    """保存或更新单个会话"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 序列化 LangChain 的消息对象为 JSON
    msg_list = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        msg_list.append({"role": role, "content": msg.content})

    messages_json = json.dumps(msg_list, ensure_ascii=False)
    created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S")

    # 使用 REPLACE 实现：如果有就更新，没有就插入
    cursor.execute('''
                   REPLACE
                   INTO sessions (session_id, title, created_at, messages)
        VALUES (?, ?, ?, ?)
                   ''', (session_id, title, created_at_str, messages_json))

    conn.commit()
    conn.close()


def load_all_sessions():
    """加载所有历史会话，转换为 st.session_state 需要的字典格式"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, title, created_at, messages FROM sessions")
    rows = cursor.fetchall()
    conn.close()

    sessions = {}
    for row in rows:
        session_id, title, created_at_str, messages_json = row
        created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")

        # 反序列化 JSON 为 LangChain 消息对象
        msg_list = json.loads(messages_json)
        messages = []
        for m in msg_list:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))

        sessions[session_id] = {
            "title": title,
            "created_at": created_at,
            "messages": messages
        }
    return sessions