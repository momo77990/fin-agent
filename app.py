import html
import json
import os
import time
from datetime import date

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

# ── 页面配置（必须是第一个 Streamlit 调用） ──────────────────────────
st.set_page_config(
    page_title="Fin-Agent | 智能金融研报助手",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# Fin-Agent\n基于 LLM Agent 的智能金融研报助手\n\n"
                 "技术栈: LangGraph + DeepSeek-V3 + RAG + yfinance + Tavily"
    },
)

# ── 延迟导入 agent 与 db（避免在 set_page_config 之前触发 Streamlit 调用） ──
from agent_graph import get_agent_app, PROVIDER_PRESETS
from db import auth as db_auth, chat_store, settings_store, init_db
from db.auth import AuthError


# ── 自定义 CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.2rem;
    max-width: 1180px;
}

[data-testid="stSidebar"] > div:first-child {
    background: #0B0F17;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.12);
    background: rgba(15, 23, 42, 0.58);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
    padding: 1rem 1.2rem;
}

.main-header {
    position: relative;
    overflow: hidden;
    background: radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 36%),
                linear-gradient(135deg, #0B1120 0%, #111827 58%, #0B1120 100%);
    border-radius: 24px;
    padding: 2.2rem 2.6rem;
    margin-bottom: 1.4rem;
    border: 1px solid rgba(148, 163, 184, 0.14);
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
}
.main-header h1 {
    color: #F8FAFC;
    font-size: 2.35rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin: 0 0 0.6rem 0;
}
.main-header p {
    color: #94A3B8;
    font-size: 1rem;
    line-height: 1.7;
    margin: 0;
    max-width: 720px;
}
.main-kicker {
    color: #2DD4BF;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

.report-card {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.96) 0%, rgba(17, 24, 39, 0.96) 100%);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    border: 1px solid rgba(148, 163, 184, 0.14);
    margin: 0.8rem 0 1rem 0;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.22);
}

.section-title {
    color: #E2E8F0;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.sidebar-brand {
    font-size: 1.35rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.03em;
    margin-bottom: 0.15rem;
}
.sidebar-subtitle {
    color: #94A3B8;
    font-size: 0.82rem;
    margin-bottom: 1rem;
}
.sidebar-user {
    color: #CBD5E1;
    font-size: 0.85rem;
    background: rgba(20, 184, 166, 0.08);
    border: 1px solid rgba(45, 212, 191, 0.18);
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.6rem;
}

[data-testid="stSidebar"] button[kind="secondary"] {
    text-align: left !important;
    font-size: 0.84rem;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background: rgba(15, 23, 42, 0.52);
    border-radius: 12px;
    transition: all 0.18s ease;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    border-color: rgba(45, 212, 191, 0.52);
    background: rgba(20, 184, 166, 0.1);
    transform: translateY(-1px);
}

.conv-row-active button[kind="secondary"] {
    border-color: rgba(45, 212, 191, 0.55) !important;
    background: rgba(20, 184, 166, 0.16) !important;
    color: #F8FAFC !important;
}

.streamlit-expanderHeader {
    font-size: 0.9rem;
    border-radius: 10px;
}

hr {
    border-color: rgba(148, 163, 184, 0.12) !important;
}

.capability-tag {
    display: inline-block;
    padding: 0.38rem 0.68rem;
    border-radius: 999px;
    font-size: 0.78rem;
    margin: 0.18rem 0.12rem;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(45, 212, 191, 0.18);
    color: #CBD5E1;
}
.process-summary {
    padding: 0.7rem 0.9rem;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.62);
    border: 1px solid rgba(45, 212, 191, 0.14);
    color: #CBD5E1;
    margin: 0.55rem 0;
    line-height: 1.65;
}
.process-meta {
    color: #94A3B8;
    font-size: 0.8rem;
    margin-bottom: 0.45rem;
}
</style>
""", unsafe_allow_html=True)


# ── 数据库初始化 ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _ensure_db():
    init_db()
    return True


try:
    _ensure_db()
except Exception as e:
    st.error(
        "无法连接数据库。请先启动 MariaDB：\n\n"
        "```\ndocker compose up -d\n```\n\n"
        f"错误详情：{e}"
    )
    st.stop()


# ── 辅助函数 ─────────────────────────────────────────────────────────

_TOOL_DISPLAY = {
    "get_stock_price":               ("股价查询",  "正在查询股价数据..."),
    "tavily_search_results_json":    ("新闻检索",  "正在搜索最新新闻..."),
    "tavily_news_search":            ("新闻检索",  "正在搜索最新新闻..."),
    "tavily_news_search_unavailable":("新闻检索",  "Tavily 不可用"),
    "search_filings":                ("研报检索",  "正在检索本地财报 / 研报..."),
    "plot_stock_chart":              ("图表生成",  "正在绘制股价走势图..."),
}


def get_tool_display(name: str):
    return _TOOL_DISPLAY.get(name, (name, f"正在执行 {name}..."))


def parse_intermediate_steps(messages):
    """从 LangGraph 消息列表中提取工具调用步骤。"""
    steps = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                steps.append({
                    "tool_name":  tc.get("name", "unknown"),
                    "tool_input": str(tc.get("args", {})),
                    "tool_output": None,
                })
        elif isinstance(msg, ToolMessage):
            for step in reversed(steps):
                if step["tool_output"] is None:
                    step["tool_output"] = msg.content
                    step["tool_artifact"] = getattr(msg, "artifact", None)
                    break
    return steps


def parse_stock_tool_output(output: str):
    """兼容旧历史：从带 PRICE_DATA_JSON 的字符串里解出价格数据。"""
    if "PRICE_DATA_JSON:" not in output:
        return None

    summary, raw_prices = output.split("PRICE_DATA_JSON:", 1)
    try:
        prices = json.loads(raw_prices.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(prices, list) or not prices:
        return None
    return {"summary": summary.strip(), "prices": prices}


def render_stock_tool_output(output: str, artifact=None):
    parsed = None
    if isinstance(artifact, dict) and isinstance(artifact.get("prices"), list) and artifact["prices"]:
        parsed = {
            "summary": (artifact.get("summary") or output or "").strip(),
            "prices": artifact["prices"],
        }
    else:
        parsed = parse_stock_tool_output(output)
    if not parsed:
        render_generic_tool_output(output)
        return

    prices_df = pd.DataFrame(parsed["prices"])
    if prices_df.empty or not {"date", "close"}.issubset(prices_df.columns):
        render_generic_tool_output(output)
        return

    prices_df["close"] = pd.to_numeric(prices_df["close"], errors="coerce")
    prices_df = prices_df.dropna(subset=["close"])
    if prices_df.empty:
        render_generic_tool_output(output)
        return

    latest_close = prices_df["close"].iloc[-1]
    start_close = prices_df["close"].iloc[0]
    change_pct = ((latest_close - start_close) / start_close) * 100
    high_close = prices_df["close"].max()
    low_close = prices_df["close"].min()

    summary_html = html.escape(parsed["summary"]).replace("\n", "<br>")
    st.markdown(
        f'<div class="process-summary">{summary_html}</div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新收盘价", f"{latest_close:.2f}")
    col2.metric("区间涨跌幅", f"{change_pct:+.2f}%")
    col3.metric("区间最高", f"{high_close:.2f}")
    col4.metric("区间最低", f"{low_close:.2f}")

    chart_df = prices_df.set_index("date")[["close"]]
    st.line_chart(chart_df, use_container_width=True)
    st.dataframe(
        prices_df.rename(columns={"date": "日期", "close": "收盘价"}),
        use_container_width=True,
        hide_index=True,
    )


def render_generic_tool_output(output: str):
    output = output.strip()
    if not output:
        st.caption("工具未返回内容。")
        return

    preview = output[:800]
    if len(output) > 800:
        preview += "\n\n..."
    preview_html = html.escape(preview).replace("\n", "<br>")
    st.markdown(f'<div class="process-summary">{preview_html}</div>', unsafe_allow_html=True)
    if len(output) > 800:
        with st.expander("查看完整工具返回", expanded=False):
            st.code(output[:3000], language="text")


def display_tool_steps(steps):
    """用 expander 展示每个工具调用的详情。"""
    if not steps:
        return
    st.markdown('<div class="section-title">分析过程</div>', unsafe_allow_html=True)
    for i, step in enumerate(steps, 1):
        label, _ = get_tool_display(step["tool_name"])
        with st.expander(f"第 {i} 步 · {label} · 已完成", expanded=False):
            tool_input_html = html.escape(step["tool_input"])
            st.markdown(
                f'<div class="process-meta">调用参数：{tool_input_html}</div>',
                unsafe_allow_html=True,
            )
            if step["tool_output"]:
                if step["tool_name"] == "get_stock_price":
                    render_stock_tool_output(step["tool_output"], step.get("tool_artifact"))
                else:
                    render_generic_tool_output(step["tool_output"])


def build_conversation_messages(history, max_turns=6):
    recent_history = history[-max_turns * 2:]
    messages = []
    for msg in recent_history:
        content = msg.get("content", "")
        if not content:
            continue
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def save_chart_for_conversation(conv_id: int) -> str | None:
    """把临时的 static/chart.png 移到按会话隔离的目录，并返回新路径。"""
    src = "static/chart.png"
    if not os.path.exists(src):
        return None
    os.makedirs("static/charts", exist_ok=True)
    dst = f"static/charts/conv{conv_id}_{int(time.time())}.png"
    try:
        os.replace(src, dst)
    except OSError:
        return None
    return dst


# ── Session State 初始化 ─────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "view" not in st.session_state:
    st.session_state.view = "chat"  # "chat" | "settings"


# ── 未登录：渲染登录/注册页 ──────────────────────────────────────────
def render_auth_page():
    st.markdown("""
    <div class="main-header">
        <div class="main-kicker">Financial Intelligence Agent</div>
        <h1>欢迎使用 Fin-Agent</h1>
        <p>请先登录或注册，您的对话历史将自动保存。</p>
    </div>
    """, unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            submitted = st.form_submit_button("登录", use_container_width=True)
            if submitted:
                try:
                    user = db_auth.authenticate_user(username, password)
                except Exception as e:
                    st.error(f"登录失败：{e}")
                else:
                    if user is None:
                        st.error("用户名或密码错误。")
                    else:
                        st.session_state.user = user
                        st.session_state.current_conv_id = None
                        st.session_state.chat_history = []
                        st.rerun()

    with register_tab:
        with st.form("register_form", clear_on_submit=False):
            username = st.text_input("用户名（3-32 位字母/数字/下划线）", key="reg_username")
            password = st.text_input("密码（6-64 位）", type="password", key="reg_password")
            password2 = st.text_input("确认密码", type="password", key="reg_password2")
            submitted = st.form_submit_button("注册并登录", use_container_width=True)
            if submitted:
                if password != password2:
                    st.error("两次输入的密码不一致。")
                else:
                    try:
                        user = db_auth.register_user(username, password)
                    except AuthError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"注册失败：{e}")
                    else:
                        st.session_state.user = user
                        st.session_state.current_conv_id = None
                        st.session_state.chat_history = []
                        st.success("注册成功，已自动登录。")
                        st.rerun()


if not st.session_state.user:
    render_auth_page()
    st.stop()


# ── 已登录：侧边栏 ──────────────────────────────────────────────────
user = st.session_state.user

with st.sidebar:
    st.markdown('<div class="sidebar-brand">Fin-Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">智能金融研报助手</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="sidebar-user">当前用户：<b>{html.escape(user["username"])}</b></div>',
        unsafe_allow_html=True,
    )
    if st.button("退出登录", use_container_width=True, key="logout_btn"):
        st.session_state.user = None
        st.session_state.current_conv_id = None
        st.session_state.chat_history = []
        st.session_state.view = "chat"
        st.rerun()

    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.button(
            "对话",
            use_container_width=True,
            key="nav_chat",
            type="primary" if st.session_state.view == "chat" else "secondary",
        ):
            st.session_state.view = "chat"
            st.rerun()
    with nav_cols[1]:
        if st.button(
            "API 设置",
            use_container_width=True,
            key="nav_settings",
            type="primary" if st.session_state.view == "settings" else "secondary",
        ):
            st.session_state.view = "settings"
            st.rerun()

    st.divider()

    # 会话管理
    st.markdown('<div class="section-title">对话历史</div>', unsafe_allow_html=True)
    if st.button("+ 新建对话", use_container_width=True, key="new_conv_btn"):
        new_id = chat_store.create_conversation(user["id"])
        st.session_state.current_conv_id = new_id
        st.session_state.chat_history = []
        st.rerun()

    conversations = chat_store.list_conversations(user["id"])
    for conv in conversations:
        is_active = conv["id"] == st.session_state.current_conv_id
        if is_active:
            st.markdown('<div class="conv-row-active">', unsafe_allow_html=True)
        cols = st.columns([0.78, 0.22])
        with cols[0]:
            title = conv["title"] or "新对话"
            if st.button(title, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.current_conv_id = conv["id"]
                st.session_state.chat_history = chat_store.get_messages(user["id"], conv["id"])
                st.rerun()
        with cols[1]:
            with st.popover("•••", use_container_width=True):
                new_title = st.text_input(
                    "重命名",
                    value=conv["title"],
                    key=f"rename_input_{conv['id']}",
                )
                if st.button("保存", key=f"rename_save_{conv['id']}", use_container_width=True):
                    if chat_store.rename_conversation(user["id"], conv["id"], new_title):
                        st.rerun()
                if st.button(
                    "删除对话",
                    key=f"delete_{conv['id']}",
                    use_container_width=True,
                    type="primary",
                ):
                    chat_store.delete_conversation(user["id"], conv["id"])
                    if st.session_state.current_conv_id == conv["id"]:
                        st.session_state.current_conv_id = None
                        st.session_state.chat_history = []
                    st.rerun()
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # 能力标签
    st.markdown('<div class="section-title">核心能力</div>', unsafe_allow_html=True)
    st.markdown("""
<span class="capability-tag">股价查询</span>
<span class="capability-tag">新闻检索</span>
<span class="capability-tag">财报分析</span>
<span class="capability-tag">工具编排</span>
""", unsafe_allow_html=True)
    st.markdown("")
    st.divider()

    # 示例指令
    st.markdown('<div class="section-title">示例指令</div>', unsafe_allow_html=True)
    _examples = [
        "分析英伟达 NVIDIA 近期股价走势和市场情况",
        "查询苹果 AAPL 最近一个月的股价数据",
        "帮我分析贵州茅台 600519.SS 近期表现",
        "搜索最新的 AI 芯片行业新闻",
        "从已上传的研报中总结英伟达数据中心业务的关键风险",
    ]
    for q in _examples:
        if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()


# ── 主区域 Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-kicker">Financial Intelligence Agent</div>
    <h1>智能金融研报助手</h1>
    <p>融合股价、新闻与研报信息，面向投资研究场景生成结构化分析报告。</p>
</div>
""", unsafe_allow_html=True)


# ── API 设置页 ───────────────────────────────────────────────────────
def render_settings_page():
    st.markdown('<div class="section-title">API 设置</div>', unsafe_allow_html=True)
    st.caption(
        "在此处配置 LLM 接口（OpenAI 兼容协议，覆盖 DeepSeek / SiliconFlow / 阿里云百炼 / Ollama / 自托管 vLLM 等）。"
        "API Key 仅保存在你登录的账户下，其他用户不可见。"
    )

    # 显示当前已索引的研报 PDF
    try:
        from rag_engine import list_indexed_documents

        indexed = list_indexed_documents()
        if indexed:
            st.info(
                "已索引的本地研报 / 财报 PDF（放置于 `data/` 目录）：\n\n"
                + "\n".join(f"- {name}" for name in indexed)
            )
        else:
            st.warning(
                "`data/` 目录下暂无 PDF。可将研报 / 财报放入该目录（容器内路径为 `/app/data`，"
                "由 docker-compose 中的 `./data` 挂载），保存设置后即可让 Agent 调用 RAG 检索。"
            )
    except Exception as exc:
        st.caption(f"无法读取 RAG 索引目录：{exc}")

    existing = settings_store.get_settings(user["id"]) or {}
    provider_keys = list(PROVIDER_PRESETS.keys())
    current_provider = existing.get("llm_provider") or "openai_compatible"
    # 兼容旧值：openai_compatible → 默认到 custom
    if current_provider not in provider_keys:
        current_provider = "custom"

    preset_index = provider_keys.index(current_provider)
    selected_provider = st.selectbox(
        "Provider 预设",
        options=provider_keys,
        index=preset_index,
        format_func=lambda k: PROVIDER_PRESETS[k]["label"],
        key="settings_provider",
        help="选择常见服务商可自动填入 base_url 与默认模型名；选『自定义』可自由输入。",
    )

    preset = PROVIDER_PRESETS[selected_provider]
    # 若用户切换了 provider，回填默认值；否则保留已存值
    default_base_url = (
        existing.get("llm_base_url")
        if existing.get("llm_provider") == selected_provider
        else preset["base_url"]
    )
    default_model = (
        existing.get("llm_model")
        if existing.get("llm_provider") == selected_provider
        else preset["model"]
    )

    with st.form("settings_form", clear_on_submit=False):
        base_url = st.text_input(
            "Base URL",
            value=default_base_url or "",
            placeholder="https://api.openai.com/v1",
            help="OpenAI 兼容的 /v1 接口地址。Ollama 本地：http://host.docker.internal:11434/v1",
        )
        model = st.text_input(
            "Model",
            value=default_model or "",
            placeholder="gpt-4o-mini / deepseek-chat / qwen-plus / Qwen/Qwen2.5-72B-Instruct",
        )
        api_key = st.text_input(
            "API Key",
            value=existing.get("llm_api_key") or "",
            type="password",
            placeholder="sk-... 留空即沿用已保存值",
        )
        embedding_model = st.text_input(
            "Embedding Model（用于 RAG 检索 PDF 研报，可选）",
            value=existing.get("embedding_model") or "",
            placeholder="text-embedding-3-small / BAAI/bge-large-zh-v1.5 / text-embedding-v4",
            help=(
                "OpenAI 兼容协议下的嵌入模型名。空着表示默认 text-embedding-3-small。"
                "注意：DeepSeek 不提供嵌入接口，使用 DeepSeek 时请改用其他 provider 的嵌入服务，或保持 data/ 为空以跳过 RAG。"
            ),
        )

        st.markdown("---")
        tavily_key = st.text_input(
            "Tavily API Key（可选，用于新闻检索）",
            value=existing.get("tavily_api_key") or "",
            type="password",
            placeholder="tvly-...",
        )

        col_save, col_clear = st.columns([0.7, 0.3])
        with col_save:
            saved = st.form_submit_button("保存配置", type="primary", use_container_width=True)
        with col_clear:
            cleared = st.form_submit_button("清除全部", use_container_width=True)

    if saved:
        if not api_key.strip() and not (existing.get("llm_api_key")):
            st.error("请填写 API Key。")
        elif not model.strip():
            st.error("请填写 Model 名称。")
        else:
            settings_store.upsert_settings(
                user["id"],
                llm_provider=selected_provider,
                llm_base_url=base_url,
                llm_api_key=api_key or existing.get("llm_api_key"),
                llm_model=model,
                embedding_model=embedding_model,
                tavily_api_key=tavily_key,
            )
            st.success("已保存。")
            st.session_state.view = "chat"
            st.rerun()

    if cleared:
        settings_store.upsert_settings(
            user["id"],
            llm_provider=selected_provider,
            llm_base_url=None,
            llm_api_key=None,
            llm_model=None,
            embedding_model=None,
            tavily_api_key=None,
        )
        st.success("已清除。")
        st.rerun()


if st.session_state.view == "settings":
    render_settings_page()
    st.stop()


# ── 对话页：先校验 API 是否已配置 ────────────────────────────────────
user_settings = settings_store.get_settings(user["id"])
if not settings_store.is_llm_configured(user_settings):
    st.warning("尚未配置 LLM API。请在左侧「API 设置」中填入 Base URL / Model / API Key 后再开始对话。")
    if st.button("前往 API 设置", type="primary"):
        st.session_state.view = "settings"
        st.rerun()
    st.stop()


# ── 聊天历史展示 ─────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if msg.get("tool_steps"):
                display_tool_steps(msg["tool_steps"])
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">深度分析报告</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)
            if msg.get("chart_path") and os.path.exists(msg["chart_path"]):
                st.image(msg["chart_path"], caption="股价走势图", use_container_width=True)
        else:
            st.markdown(msg["content"])


# ── 聊天输入 ─────────────────────────────────────────────────────────
user_input = st.chat_input("请输入金融分析指令，例如：分析英伟达 NVIDIA 近期情况")

if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input:
    # 0) 确保有 current_conv_id
    if not st.session_state.current_conv_id:
        st.session_state.current_conv_id = chat_store.create_conversation(user["id"])
    conv_id = st.session_state.current_conv_id

    # 1) 保存并展示用户消息
    conversation_messages = build_conversation_messages(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    chat_store.append_message(user["id"], conv_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) 调用 Agent 并流式展示回复
    with st.chat_message("assistant"):
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">深度分析报告</div>', unsafe_allow_html=True)
        report_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        with st.status("AI Agent 正在分析...", expanded=True) as status:
            st.write("初始化 Agent，准备调用工具...")
            all_messages = []
            streamed_response = ""
            tool_steps = []

            try:
                today = date.today().isoformat()
                system_prompt = (
                    f"今天是 {today}。你是一个金融研报助手。"
                    "当用户询问股价、股票走势、历史价格、最近一个月表现或提供股票代码时，必须先调用股价查询工具获取数据，不能直接回答无法查询。"
                    "当用户询问最新、今天、近期、新闻、实时信息或联网搜索时，必须调用新闻检索工具，不能只依赖模型自身知识。"
                    "当用户询问公司基本面、业务结构、风险因素、管理层讨论、招股书细节或 PDF 研报内的具体内容时，调用 search_filings 检索本地索引的财报/研报 PDF；"
                    "若该工具返回『不可用』或『未找到』，再回退到模型知识或新闻检索。"
                    "最终报告必须使用结构化 Markdown，只总结关键指标、趋势和结论，不要粘贴完整原始数据表。"
                    "工具调用过程由前端的分析过程区域展示，最终报告只输出面向用户的分析结论。"
                )
                agent_messages = [
                    SystemMessage(content=system_prompt),
                    *conversation_messages,
                    HumanMessage(content=user_input),
                ]
                agent_app = get_agent_app(user_settings)
                stream = agent_app.stream(
                    {"messages": agent_messages},
                    stream_mode=["messages", "updates"],
                )

                for mode, event in stream:
                    if mode == "messages":
                        chunk, _metadata = event
                        # 只接受 LLM 产生的 token chunk;ToolMessage 等其他类型会把工具原始返回(如 RAG 片段)
                        # 也累加进最终报告,必须过滤掉
                        if not isinstance(chunk, AIMessageChunk):
                            continue
                        content = getattr(chunk, "content", "")
                        if isinstance(content, list):
                            content = "".join(
                                part.get("text", "") if isinstance(part, dict) else str(part)
                                for part in content
                            )
                        elif not isinstance(content, str):
                            content = str(content) if content else ""

                        if content:
                            streamed_response += content
                            report_placeholder.markdown(streamed_response + "▌")

                    elif mode == "updates":
                        for node_name, update in event.items():
                            new_messages = update.get("messages", [])
                            all_messages.extend(new_messages)

                            if node_name == "agent":
                                for msg in new_messages:
                                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                                        for tool_call in msg.tool_calls:
                                            _label, running_text = get_tool_display(
                                                tool_call.get("name", "unknown")
                                            )
                                            st.write(running_text)
                            elif node_name == "tools":
                                for msg in new_messages:
                                    if isinstance(msg, ToolMessage):
                                        st.write("工具调用完成")
            except Exception as e:
                st.error(f"Agent 执行出错：{e}")
                status.update(label="分析失败", state="error")
                st.stop()

            final_response = streamed_response or (
                all_messages[-1].content if all_messages else "未获取到分析结果。"
            )
            report_placeholder.markdown(final_response)
            tool_steps = parse_intermediate_steps(all_messages)

            for step in tool_steps:
                label, _ = get_tool_display(step["tool_name"])
                st.write(f"{label} 完成")

            status.update(label="分析完成", state="complete", expanded=False)

        # 3) 展示工具调用详情
        if tool_steps:
            display_tool_steps(tool_steps)

        # 4) 处理图表：把临时文件迁到按会话隔离的目录
        chart_path = save_chart_for_conversation(conv_id)
        if chart_path:
            st.image(chart_path, caption="股价走势图", use_container_width=True)

        # 5) 保存助手消息到内存与数据库
        assistant_entry = {
            "role": "assistant",
            "content": final_response,
            "tool_steps": tool_steps,
            "chart_path": chart_path,
        }
        st.session_state.chat_history.append(assistant_entry)
        chat_store.append_message(
            user["id"],
            conv_id,
            "assistant",
            final_response,
            tool_steps=tool_steps,
            chart_path=chart_path,
        )
