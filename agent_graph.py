"""按用户 LLM 配置构建 LangGraph Agent。"""

from functools import lru_cache
from typing import Optional, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools.news_tools import get_news_tool
from tools.rag_tools import build_rag_tool
from tools.stock_tools import get_stock_price


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 各 provider 的默认 base_url，前端选择 preset 时回填到表单
PROVIDER_PRESETS = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "siliconflow": {
        "label": "SiliconFlow",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-72B-Instruct",
    },
    "dashscope": {
        "label": "阿里云百炼 (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "ollama": {
        "label": "Ollama (本地)",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
    },
    "custom": {
        "label": "自定义 (OpenAI 兼容)",
        "base_url": "",
        "model": "",
    },
}


@lru_cache(maxsize=64)
def _build_agent(
    base_url: Optional[str],
    api_key: str,
    model: str,
    embedding_model: Optional[str],
    tavily_api_key: Optional[str],
):
    """根据 LLM 配置构建并编译一个 StateGraph。结果按参数缓存。"""
    llm_kwargs = {"model": model, "temperature": 0, "streaming": True, "api_key": api_key}
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm = ChatOpenAI(**llm_kwargs)

    tools = [
        get_stock_price,
        get_news_tool(tavily_api_key),
        build_rag_tool(
            api_key=api_key,
            base_url=base_url,
            embedding_model=embedding_model,
        ),
    ]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def get_agent_app(config: dict):
    """对外入口：根据用户配置返回（缓存的）已编译图。"""
    return _build_agent(
        base_url=(config.get("llm_base_url") or None),
        api_key=config["llm_api_key"],
        model=config["llm_model"],
        embedding_model=(config.get("embedding_model") or None),
        tavily_api_key=(config.get("tavily_api_key") or None),
    )
