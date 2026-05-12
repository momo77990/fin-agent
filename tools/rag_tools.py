"""金融文档检索工具（RAG）。

由 agent_graph 在构图时按用户配置注入：embedding 凭据 + 模型名。
工具在调用时按 retriever 召回 top-k 文档片段，拼成上下文返回给 LLM。
"""

from typing import Optional

from langchain.tools import tool

from rag_engine import DEFAULT_EMBEDDING_MODEL, RagUnavailable, get_retriever


def build_rag_tool(
    api_key: Optional[str],
    base_url: Optional[str] = None,
    embedding_model: Optional[str] = None,
    data_dir: str = "data",
    k: int = 4,
):
    """构造一个名为 `search_filings` 的 @tool。

    若没有提供 api_key、目录为空或嵌入失败，仍返回一个可调用的「降级」工具，
    使 LLM 收到清晰的错误说明而不是 500。
    """
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    @tool
    def search_filings(query: str) -> str:
        """从已索引的金融研报 / 财报 / 招股书 PDF 中检索与查询相关的段落。

        适用于用户询问公司基本面、业务结构、风险因素、管理层讨论与分析、
        往年财务数据细节、或 PDF 文件内的具体内容时。

        Args:
            query: 用自然语言描述要查的内容，例如「英伟达数据中心业务收入构成」。

        Returns:
            拼接好的文档片段及其来源页码；若无可用文档，返回提示信息。
        """
        if not api_key:
            return "RAG 工具未启用：当前用户未配置 LLM API Key，无法初始化嵌入模型。"

        try:
            retriever = get_retriever(
                api_key=api_key,
                base_url=base_url,
                embedding_model=embedding_model,
                data_dir=data_dir,
                k=k,
            )
        except RagUnavailable as exc:
            return f"RAG 工具不可用：{exc}"
        except Exception as exc:
            return f"RAG 工具初始化失败：{exc}"

        try:
            docs = retriever.invoke(query)
        except Exception as exc:
            return f"检索失败：{exc}"

        if not docs:
            return f"未在已索引文档中找到与「{query}」相关的内容。"

        chunks = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            content = doc.page_content.strip().replace("\n\n", "\n")
            chunks.append(f"[片段 {i} · 来源: {source} · 第 {page} 页]\n{content}")
        return "\n\n---\n\n".join(chunks)

    return search_filings
