"""快速测试 LLM 连通性，请在你的 conda 环境里运行: python test_llm.py"""
from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
api_key = dashscope_api_key or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
if dashscope_api_key and not base_url:
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = os.getenv("DASHSCOPE_MODEL") or os.getenv("OPENAI_MODEL", "qwen3-plus")

print(f"base_url = {base_url}")
print(f"model    = {model}")
print(f"api_key  = {api_key[:8]}...{api_key[-4:]}" if api_key else "api_key  = 未设置")
print()

llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, base_url=base_url)

# 测试 1: 纯文本调用（不绑定 tools）
print("--- 测试 1: 纯文本调用 ---")
try:
    resp = llm.invoke([HumanMessage(content="你好，请回复OK")])
    print("成功:", resp.content)
except Exception as e:
    print("失败:", type(e).__name__, str(e)[:300])

# 测试 2: 绑定 tools 调用
print("\n--- 测试 2: 绑定 tools 调用 ---")
try:
    from tools.stock_tools import get_stock_price
    llm_with_tools = llm.bind_tools([get_stock_price])
    resp2 = llm_with_tools.invoke([HumanMessage(content="你好，请回复OK")])
    print("成功:", resp2.content)
except Exception as e:
    print("失败:", type(e).__name__, str(e)[:300])
