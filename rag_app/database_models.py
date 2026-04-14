from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ===================== 数据库连接配置 =====================
# Windows 下的 SQLite 本地数据库文件
SQLALCHEMY_DATABASE_URL = "sqlite:///./deepblue_rag.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ===================== 1. 账号信息表 =====================
class User(Base):
    """
    用于记录账号信息
    管理员账号后端标识 role 为 1，普通用户为 0
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False) # 例如: "admin"
    password_hash = Column(String, nullable=False)                    # 密码哈希值
    role = Column(Integer, default=0, nullable=False)                 # 1: 管理员, 0: 普通用户

    # 关联设置和历史会话
    settings = relationship("UserSettings", back_populates="user", uselist=False)
    chat_sessions = relationship("ChatSession", back_populates="user")
    setting_histories = relationship("SettingHistory", back_populates="user")


# ===================== 2. 历史记录表 =====================
class ChatSession(Base):
    """
    用于记录 Chat 里的会话（左侧边栏列表）
    """
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True) # 会话的唯一ID (如时间戳或UUID)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关联用户和消息
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """
    用于记录 Chat 里的具体消息记录 (User 和 AI 的一问一答)
    """
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)            # "user" 或 "ai"
    content = Column(Text, nullable=False)           # 实际显示的文本内容
    search_process = Column(Text, nullable=True)     # AI 专属：检索过程描述
    citations = Column(Text, nullable=True)          # AI 专属：引用的文档 (可存 JSON 字符串)
    kg_triplets = Column(Text, nullable=True)        # AI 专属：图谱三元组 (可存 JSON 字符串)
    extra_data = Column(Text, nullable=True)         # AI 专属：其他附加字段 (可存 JSON 字符串)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


# ===================== 3. 个人信息表 =====================
class UserSettings(Base):
    """
    用于记录 Setting 里的个人信息和系统偏好
    """
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"), unique=True, nullable=False)
    
    # 个人信息
    nickname = Column(String, default="Captain")
    avatar = Column(Text, default="")
    imo = Column(String, default="")
    email = Column(String, default="")
    emergency = Column(String, default="")
    
    # 系统偏好
    theme = Column(String, default="dark")
    font_size = Column(Integer, default=16)
    notify = Column(Boolean, default=True)
    auto_save = Column(Boolean, default=True)
    
    # AI 与知识图谱
    graph_on = Column(Boolean, default=True)
    offline_on = Column(Boolean, default=False)
    stream_speed = Column(Integer, default=50)
    db_path = Column(String, default="/mnt/data/local_kb")
    model = Column(String, default="hybrid")
    
    # 隐私与安全
    retention = Column(String, default="30")

    user = relationship("User", back_populates="settings")


class SettingHistory(Base):
    """
    用于记录 Setting 修改历史（审计与回溯）
    """

    __tablename__ = "setting_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    changed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    changed_by = Column(String, default="system")
    payload = Column(Text, nullable=False)  # 保存完整设置快照 JSON

    user = relationship("User", back_populates="setting_histories")


def _ensure_sqlite_column(table_name: str, column_name: str, column_sql: str) -> None:
    with engine.begin() as conn:
        columns = conn.execute(text(f"PRAGMA table_info('{table_name}')")).fetchall()
        existing = {col[1] for col in columns}
        if column_name not in existing:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))

# ===================== 初始化脚本 =====================
def init_db():
    """在应用启动时调用此函数以创建表"""
    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_column("user_settings", "avatar", "TEXT DEFAULT ''")
    _ensure_sqlite_column("chat_messages", "kg_triplets", "TEXT")
    _ensure_sqlite_column("chat_messages", "extra_data", "TEXT")

if __name__ == "__main__":
    init_db()
    print("数据库架构已成功创建！")
