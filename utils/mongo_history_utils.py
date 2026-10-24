import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient

load_dotenv()


# ====================MongoDB历史对话工具====================
class HistoryMongoTool:
    def __init__(self):
        self.mongo_url = os.getenv("MONGO_URL") or "mongodb://localhost:27017"
        self.db_name = os.getenv("MONGO_DB_NAME") or "kb001"
        self.client = None
        self.db = None
        self.chat_message = None

    def connect(self):
        """按需连接MongoDB；连接断开后允许自动重连。"""
        if self.client is not None:
            try:
                self.client.admin.command("ping")
                return self
            except Exception:
                self.close()

        client = MongoClient(
            self.mongo_url,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
            socketTimeoutMS=5000,
        )
        try:
            client.admin.command("ping")
            db = client[self.db_name]
            chat_message = db["chat_message"]
            chat_message.create_index([("session_id", ASCENDING), ("ts", ASCENDING)])

            self.client = client
            self.db = db
            self.chat_message = chat_message
            logging.info(
                "Successfully connected to MongoDB %s %s",
                self.db_name,
                self.mongo_url,
            )
            return self
        except Exception as exc:
            client.close()
            logging.error("Failed to connect to MongoDB: %s", exc)
            raise ConnectionError(f"MongoDB不可用: {self.mongo_url}") from exc

    def close(self):
        if self.client is not None:
            self.client.close()
        self.client = None
        self.db = None
        self.chat_message = None


# 导入模块时只创建对象，不连接；真正读写历史时再连接。
_history_mongo_tool = HistoryMongoTool()


def get_history_mongo_tool() -> HistoryMongoTool:
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool.connect()


def clear_history(session_id) -> int:
    try:
        mongo_tool = get_history_mongo_tool()
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info(
            "Deleted %s messages for session %s",
            result.deleted_count,
            session_id,
        )
        return result.deleted_count
    except Exception as exc:
        logging.error("Failed to delete messages for session %s: %s", session_id, exc)
        return 0


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: list[str] = None,
    image_urls: list[str] = None,
    message_id: str = None,
) -> str:
    document = {
        "session_id": session_id,
        "ts": datetime.now().timestamp(),
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names or [],
        "image_urls": image_urls or [],
    }
    mongo_tool = get_history_mongo_tool()
    if message_id:
        mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
            upsert=True,
        )
        return message_id

    result = mongo_tool.chat_message.insert_one(document)
    return result.inserted_id


def update_message_item_names(ids: List[str], item_names: List[str] = None) -> int:
    try:
        mongo_tool = get_history_mongo_tool()
        object_ids = [ObjectId(item_id) for item_id in ids]
        result = mongo_tool.chat_message.update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"item_names": item_names or []}},
        )
        logging.info("更新了%s条记录，物品名字为%s", result.modified_count, item_names)
        return result.modified_count
    except Exception as exc:
        logging.error("更新对话历史失败: %s", exc)
        return 0


def get_recent_messages(
    session_id: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    try:
        mongo_tool = get_history_mongo_tool()
        cursor = (
            mongo_tool.chat_message.find({"session_id": session_id})
            .sort([("ts", ASCENDING)])
            .limit(limit)
        )
        return list(cursor)
    except Exception as exc:
        logging.error("Error getting recent messages: %s", exc)
        return []


if __name__ == "__main__":
    sid = "test_session_002"
    save_chat_message(sid, "user", "你好，有烫金机吗？")
    save_chat_message(sid, "assistant", "你好！请问你想询问哪个型号？")
    save_chat_message(sid, "user", "brother的HAK180烫金机")
    save_chat_message(sid, "assistant", "有的")
    print("MongoDB 测试消息写入成功")
