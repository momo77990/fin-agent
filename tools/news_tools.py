"""新闻搜索工具（兼容不同 langchain-community 版本，并支持运行时传入 API key）。"""

from typing import Optional

from langchain.tools import tool


def get_news_tool(tavily_api_key: Optional[str] = None):
    """返回可供 LangGraph 绑定的 Tavily 搜索工具。

    若显式传入 tavily_api_key，则优先使用；否则回退到 TAVILY_API_KEY 环境变量。
    """
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        kwargs = {"max_results": 3}
        if tavily_api_key:
            kwargs["tavily_api_key"] = tavily_api_key
        return TavilySearchResults(**kwargs)
    except Exception:
        try:
            from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

            wrapper_kwargs = {
                "max_results": 3,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
            }
            if tavily_api_key:
                wrapper_kwargs["tavily_api_key"] = tavily_api_key
            tavily_client = TavilySearchAPIWrapper(**wrapper_kwargs)

            @tool
            def tavily_news_search(query: str) -> str:
                """搜索并返回相关新闻摘要。"""
                return tavily_client.run(query)

            return tavily_news_search
        except Exception:
            @tool
            def tavily_news_search_unavailable(query: str) -> str:
                """当 Tavily 依赖不可用时返回提示信息。"""
                return (
                    f"Tavily 工具不可用，当前查询为：{query}。"
                    "请在「API 设置」中填入 Tavily API Key，"
                    "或在当前 Python 环境安装 langchain-community 和 tavily-python。"
                )

            return tavily_news_search_unavailable
