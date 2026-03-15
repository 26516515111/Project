from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
import json
import os
from langchain_core.messages import messages_from_dict, message_to_dict, BaseMessage

chat_history_record = {}
# 存储于本地
def get_history(user_id):
   return FileChatMessageHistory(user_id,"./history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_dir):
        self.session_id = session_id
        self.storage_dir = storage_dir

        self.file_path = os.path.join(self.storage_dir, f"{self.session_id}.json")

        # 创建的文件夹路径
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # 创建的是文件路径
        # os.makedirs(self.file_path,exist_ok=True)

    # 写入
    def add_message(self, message: BaseMessage) -> None:
        all_messages = list(self.messages)
        all_messages.append(message)

        messages_dict = [message_to_dict(message) for message in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(messages_dict, f)

    # 读取
    @property
    def messages(self) -> Sequence[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                message_dict = json.load(f)
                return messages_from_dict(message_dict)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)

#
#
# # 运行实例：
# if __name__ == "__main__":
#     config = {
#         "configurable": {
#             "session_id": "user1"
#         }
#     }
#     # for chunk in conversation_chain.stream({"question":"你是小明"},config=config):
#     #     print(chunk,end="",flush=True)
#
#     for chunk in conversation_chain.stream({"question": "你是谁？请介绍一下你自己"}, config=config):
#         print(chunk, end="", flush=True)
#
    # def add_message(self,message)->None:
