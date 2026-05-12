"""RAG 引擎：扫描 data/ 下所有 PDF 并构建 FAISS 向量库。

设计要点：
- 不再硬编码单个 PDF；遍历目录获得文档集合
- 由调用方注入 API 配置（base_url / api_key / embedding_model）以支持多用户多 provider
- 结果按「目录指纹 + 配置」缓存，避免每次 Agent 调用重新嵌入
- 目录为空或嵌入失败时抛出明确异常，由上层（rag_tool）转化为对 LLM 友好的提示
"""

import os
from functools import lru_cache
from glob import glob
from typing import List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class RagUnavailable(Exception):
    """RAG 不可用（无文档 / 嵌入失败 等可对用户解释的情况）。"""


def _data_dir_signature(data_dir: str) -> Tuple[Tuple[str, float, int], ...]:
    """以 (path, mtime, size) 为目录指纹，用于缓存失效。"""
    if not os.path.isdir(data_dir):
        return tuple()
    items = []
    for path in sorted(glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)):
        try:
            stat = os.stat(path)
            items.append((path, stat.st_mtime, stat.st_size))
        except OSError:
            continue
    return tuple(items)


@lru_cache(maxsize=8)
def _build_retriever_cached(
    signature: Tuple[Tuple[str, float, int], ...],
    base_url: Optional[str],
    api_key: str,
    embedding_model: str,
    k: int,
):
    if not signature:
        raise RagUnavailable("data/ 目录下没有可索引的 PDF。请将研报或财报 PDF 放入该目录后重试。")

    docs = []
    for path, _, _ in signature:
        try:
            docs.extend(PyPDFLoader(path).load())
        except Exception as exc:
            # 单个 PDF 解析失败时跳过，但不中断整体
            print(f"[rag_engine] 跳过 {path}: {exc}")
    if not docs:
        raise RagUnavailable("data/ 中的 PDF 解析均失败，请检查文件是否损坏。")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embedding_kwargs = {"model": embedding_model, "api_key": api_key}
    if base_url:
        embedding_kwargs["base_url"] = base_url

    try:
        embeddings = OpenAIEmbeddings(**embedding_kwargs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    except Exception as exc:
        raise RagUnavailable(
            f"构建向量索引失败：{exc}。请确认当前 provider 支持嵌入模型『{embedding_model}』。"
        )

    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_retriever(
    api_key: str,
    base_url: Optional[str] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    data_dir: str = DEFAULT_DATA_DIR,
    k: int = 4,
):
    """对外入口：返回（缓存的）retriever。"""
    signature = _data_dir_signature(data_dir)
    return _build_retriever_cached(signature, base_url, api_key, embedding_model, k)


def list_indexed_documents(data_dir: str = DEFAULT_DATA_DIR) -> List[str]:
    return [os.path.basename(p) for p, _, _ in _data_dir_signature(data_dir)]
