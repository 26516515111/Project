from typing import List, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter


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
                    for i, chunk in enumerate(splitter.split_text(page_text)):
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
            for i, chunk in enumerate(splitter.split_text(d["text"])):
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
        return splitter.split_text(text)

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
