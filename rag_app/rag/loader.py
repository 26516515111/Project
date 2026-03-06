import os
from typing import List


def load_documents(docs_dir: str) -> List[dict]:
    docs: List[dict] = []
    if not os.path.isdir(docs_dir):
        return docs
    for root, _, files in os.walk(docs_dir):
        for name in files:
            if not name.lower().endswith((".txt", ".md")):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                with open(path, "r", encoding="gb18030", errors="ignore") as f:
                    text = f.read().strip()
            if not text:
                continue
            docs.append({"id": path, "text": text, "source": name})
    return docs
