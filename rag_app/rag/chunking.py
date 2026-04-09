from typing import List, Optional, Union
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import SETTINGS


class TextChunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Optional[List[str]] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", "，", ",", ""]

    def split_documents(self, docs: List[dict]) -> List[dict]:
        """将文档切分为较小chunk以便向量检索。

        Args:
            docs: 原始文档列表。

        Returns:
            List[dict]: 切分后的chunk列表。
        """
        if not docs:
            return []
        if "chunk_id" in docs[0]:
            return docs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        chunks: List[dict] = []
        for d in docs:
            if "pages" in d and isinstance(d["pages"], list):
                for page in d["pages"]:
                    page_text = str(page.get("text", "")).strip()
                    if not page_text:
                        continue
                    page_num = page.get("page")
                    for i, chunk in enumerate(
                        self._split_paragraphs(page_text, splitter)
                    ):
                        chunk_id = f"{d['id']}::page_{page_num}::chunk_{i}"
                        source = d.get("source", "")
                        if page_num is not None:
                            source = (
                                f"{source}#p{page_num}" if source else f"p{page_num}"
                            )
                        chunks.append(
                            {
                                "doc_id": chunk_id,
                                "chunk_id": chunk_id,
                                "chunk_index": i,
                                "text": chunk,
                                "source": source,
                                "page": page_num,
                            }
                        )
                continue
            for i, chunk in enumerate(self._split_paragraphs(d["text"], splitter)):
                chunk_id = f"{d['id']}::chunk_{i}"
                chunks.append(
                    {
                        "doc_id": chunk_id,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "text": chunk,
                        "source": d["source"],
                    }
                )
        return chunks

    def split_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        return self._split_paragraphs(text, splitter)

    def _split_paragraphs(
        self,
        text: str,
        splitter: RecursiveCharacterTextSplitter,
    ) -> List[str]:
        value = str(text or "").strip()
        if not value:
            return []
        heading_chunks = self._split_by_headings(value, splitter)
        if heading_chunks:
            return heading_chunks
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", value) if p.strip()]
        if not paragraphs:
            return []
        paragraphs = self._merge_heading_paragraphs(paragraphs)
        chunks: List[str] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                chunks.append(para)
                continue
            chunks.extend(splitter.split_text(para))
        return chunks

    def _split_by_headings(
        self,
        text: str,
        splitter: RecursiveCharacterTextSplitter,
    ) -> List[str]:
        lines = [line.rstrip() for line in str(text or "").splitlines()]
        
        # 匹配 [H1], [H2] 或 Markdown 标题 #, ## 等，以及传统的 1.1, 1.2 标题
        blocks: List[List[str]] = []
        current: List[str] = []
        current_level: int | None = None
        for line in lines:
            level = None
            
            # 1. 尝试匹配 [H1] 标签
            h_match = re.match(r"^\s*\[H(\d)\]\s*", line)
            # 2. 尝试匹配 Markdown 标题
            md_match = re.match(r"^\s*(#+)\s+", line)
            # 3. 尝试匹配数字分级标题 (例如: 1.1, 1.2.3)
            num_match = re.match(r"^\s*(?:\[[A-Za-z0-9]+\]\s*)?(\d+(?:\.\d+)+)\s+.+", line)

            if h_match:
                level = int(h_match.group(1))
            elif md_match:
                level = len(md_match.group(1))
            elif num_match:
                level = len(num_match.group(1).split("."))

            if level is not None:
                if not SETTINGS.heading_merge_enabled:
                    if current:
                        blocks.append(current)
                    current = [line]
                    current_level = None
                    continue
                merge_level = SETTINGS.heading_merge_level
                if level <= merge_level:
                    if current:
                        blocks.append(current)
                    current = [line]
                    current_level = level
                    continue
                if current_level is None:
                    current = [line]
                    current_level = level
                    continue
            if current or line.strip():
                current.append(line)
        if current:
            blocks.append(current)

        if len(blocks) <= 1:
            return []

        chunks: List[str] = []
        for block in blocks:
            content = "\n".join([l for l in block if l.strip()]).strip()
            if not content:
                continue
            if len(content) <= self.chunk_size:
                chunks.append(content)
                continue
            chunks.extend(splitter.split_text(content))
        return chunks

    def _merge_heading_paragraphs(self, paragraphs: List[str]) -> List[str]:
        merged: List[str] = []
        buffer: List[str] = []

        def flush() -> None:
            if buffer:
                merged.append("\n".join(buffer).strip())
                buffer.clear()

        for para in paragraphs:
            if self._is_heading_paragraph(para):
                flush()
                buffer.append(para.strip())
                continue
            if buffer:
                buffer.append(para.strip())
            else:
                merged.append(para.strip())

        flush()
        return [part for part in merged if part]

    def _is_heading_paragraph(self, paragraph: str) -> bool:
        text = paragraph.strip()
        if not text:
            return False
        first_line = text.splitlines()[0].strip()
        if re.match(r"^\[H\d\]\s+", first_line):
            return True
        if re.match(r"^\[[A-Za-z]{2,8}\d+\]\s*", first_line):
            return True
        if len(first_line) > 40:
            return False
        if re.match(r"^第[一二三四五六七八九十百千0-9]+[章节篇部卷集].*", first_line):
            return True
        if re.match(r"^[一二三四五六七八九十]+[、.．\)]", first_line):
            return True
        if re.match(r"^\d+(\.\d+)+(\s+|[、.．\)])", first_line):
            return True
        if re.match(r"^\d+(\s+|[、.．\)])", first_line):
            return True
        if re.match(r"^[\(（]?[一二三四五六七八九十0-9]+[\)）][、.．]?", first_line):
            return True
        if re.match(r"^\d+[）)]", first_line):
            return True
        if re.match(r"^[A-Z][A-Za-z0-9\s\-/]{0,30}$", first_line):
            return True
        if len(first_line) <= 20 and re.match(
            r"^[\u4e00-\u9fffA-Za-z0-9\s]+$", first_line
        ):
            if not re.search(r"[。！？：:，,]", first_line):
                return True
        return False

    @classmethod
    def from_settings(cls, settings) -> "TextChunker":
        return cls(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=settings.separators,
        )


class LanggraphyChunkerAdapter:
    def __init__(self, chunker: TextChunker) -> None:
        self.chunker = chunker

    def split(self, data: Union[str, List[dict]]) -> Union[List[str], List[dict]]:
        if isinstance(data, str):
            return self.chunker.split_text(data)
        return self.chunker.split_documents(data)
