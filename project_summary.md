# 智能金融研报助手 (Fin-Agent) — 项目技术总结

> 适用于保研简历项目经历撰写参考

---

## 一、项目简介

基于大语言模型（LLM）的智能金融研报助手，采用 **Agent 架构**自主完成股价查询、新闻检索、金融 PDF 报告阅读与分析，最终生成结构化的深度研究报告。系统前端使用 Streamlit 构建交互式 Web 界面，用户输入自然语言指令即可获得多源数据融合的金融分析结果。

---

## 二、系统架构

```
用户输入 (自然语言)
      │
      ▼
┌─────────────┐
│  Streamlit   │  ← 前端交互层
│   Web UI     │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│     LangGraph StateGraph         │  ← Agent 编排层
│  ┌────────┐    ┌───────────┐    │
│  │ Agent  │◄──►│   Tools   │    │
│  │ Node   │    │   Node    │    │
│  └────────┘    └───────────┘    │
│   (LLM推理)    (工具执行)        │
└──────────────────────────────────┘
       │                │
       ▼                ▼
┌────────────┐  ┌──────────────┐  ┌──────────────┐
│ DeepSeek-V3│  │ Yahoo Finance│  │ Tavily Search│
│  (SiliconFlow)│  │  (yfinance)  │  │  (新闻检索)   │
└────────────┘  └──────────────┘  └──────────────┘
                                          │
                        ┌─────────────────┘
                        ▼
                ┌──────────────┐
                │  RAG 引擎     │
                │ (FAISS+PyPDF)│
                └──────────────┘
```

---

## 三、核心技术点与背后知识

### 1. ReAct Agent 推理框架

**技术实现：** 使用 LangGraph 构建有状态的 Agent 循环图（StateGraph），包含 Agent 节点和 Tools 节点，通过条件边（conditional edge）实现 LLM 自主决策——判断是否需要调用工具、调用哪个工具、何时终止推理。

**背后知识：**
- **ReAct (Reasoning + Acting) 范式**：源自 Yao et al. (2023) 的论文，核心思想是让 LLM 交替进行"推理（Thought）"和"行动（Action）"，通过观察工具返回结果不断迭代，直到获得足够信息生成最终答案。
- **有限状态机（FSM）**：LangGraph 的 StateGraph 本质上是一个有向图/状态机，节点是计算单元，边定义控制流转换。
- **消息状态累积**：使用 `add_messages` reducer 维护完整的对话历史（包括工具调用和返回），确保 Agent 在多轮推理中具备上下文感知能力。

### 2. LLM Tool Calling（函数调用）机制

**技术实现：** 通过 LangChain 的 `@tool` 装饰器将 Python 函数封装为标准化工具，使用 `bind_tools()` 将工具绑定到 LLM，模型根据用户意图自主选择调用哪些工具。

**背后知识：**
- **Function Calling / Tool Use**：LLM 根据工具的函数签名（名称、参数类型、描述）在推理时生成结构化的工具调用请求（JSON），而非直接输出自然语言，实现了 LLM 与外部系统的程序化交互。
- **工具抽象与插件化设计**：每个工具是独立的模块，遵循统一接口，可随时增删，体现了面向接口编程和开闭原则（OCP）。

### 3. RAG（检索增强生成）流水线

**技术实现：** 使用 PyPDFLoader 加载金融 PDF 报告 → RecursiveCharacterTextSplitter 分块（chunk_size=1000, overlap=200）→ OpenAI Embeddings 向量化 → FAISS 构建向量索引 → 相似度检索。

**背后知识：**
- **RAG (Retrieval-Augmented Generation)**：Lewis et al. (2020) 提出的范式，通过外部知识检索增强 LLM 的生成能力，解决 LLM 知识截止日期和幻觉（hallucination）问题。
- **文本分块策略**：递归字符分割器按字符数切分文档，保留 200 字符重叠区域以避免语义断裂，是工程实践中平衡检索精度与上下文完整性的常用方案。
- **向量嵌入（Embedding）**：将文本映射到高维向量空间，语义相似的文本在空间中距离更近。
- **FAISS 近似最近邻搜索**：Facebook 开源的向量相似度搜索库，支持高效的 ANN（Approximate Nearest Neighbor）检索，适用于大规模向量数据库场景。
- **向量数据库**：以向量为索引的新型数据存储范式，是 LLM 应用中的核心基础设施。

### 4. 多源异构数据融合

**技术实现：**
- **股价数据**：通过 yfinance 调用 Yahoo Finance API，获取美股及 A 股（如 `NVDA`、`600519.SS`）最近一个月的历史收盘价。
- **新闻数据**：通过 Tavily Search API（AI 优化的搜索引擎）获取实时金融新闻摘要。
- **文档数据**：通过 RAG 引擎对 PDF 财报进行语义检索。
- LLM Agent 将上述三类异构数据源的返回结果综合推理，生成统一的分析报告。

**背后知识：**
- **异构数据融合**：不同格式（时序数据、非结构化文本、PDF 文档）的数据需要统一表示后才能被 LLM 处理，体现了信息检索与自然语言处理的结合。
- **API 集成与数据工程**：涉及 RESTful API 调用、数据格式转换（JSON/DataFrame/String）、异常处理等工程实践。

### 5. 优雅降级（Graceful Degradation）策略

**技术实现：** 新闻搜索工具采用三级降级策略——优先使用新版 `TavilySearchResults`，失败后回退至旧版 `TavilySearchAPIWrapper`，最终回退至提示用户安装依赖的 stub 工具。

**背后知识：**
- **防御性编程**：面对依赖版本不一致的问题，通过多级 try-except 和工厂模式实现运行时自适应，保证系统在不同环境下均可运行。
- **工厂方法模式**：`get_news_tool()` 函数根据运行时环境动态返回不同的工具实例，是经典的工厂方法设计模式应用。

### 6. 大语言模型工程实践

**技术实现：**
- 使用 DeepSeek-V3（国产开源大模型）通过 SiliconFlow 的 OpenAI 兼容 API 接口调用。
- 设置 `temperature=0` 保证金融分析输出的确定性和可复现性。
- 通过环境变量（`.env` + `python-dotenv`）管理 API 密钥和模型配置，实现配置与代码分离。

**背后知识：**
- **OpenAI 兼容 API 协议**：已成为 LLM 服务的事实标准接口，不同模型提供商（DeepSeek、SiliconFlow 等）均实现该协议，使得模型可以即插即用地切换。
- **Temperature 参数**：控制 LLM 输出的随机性，`temperature=0` 采用贪心解码（greedy decoding），适合需要确定性输出的场景。
- **配置外部化（Twelve-Factor App）**：敏感信息和配置项通过环境变量注入，是云原生应用的最佳实践之一。

### 7. 数据可视化

**技术实现：** 使用 Matplotlib 生成股价走势折线图，支持标记点、网格线，保存为 PNG 图片后在 Streamlit 前端展示。

**背后知识：**
- **数据可视化**：将时序股价数据转化为直观的折线图，辅助用户理解价格趋势。
- **前后端数据流**：后端生成静态图片资源，前端通过路径引用展示，是轻量级 Web 应用的常见模式。

---

## 四、技术栈总览

| 层次 | 技术 | 用途 |
|------|------|------|
| 大模型 | DeepSeek-V3 (via SiliconFlow) | 自然语言推理与生成 |
| Agent 框架 | LangGraph + LangChain | Agent 编排、工具调用、状态管理 |
| 向量数据库 | FAISS (faiss-cpu) | 文档向量索引与相似度检索 |
| 文档处理 | PyPDF + RecursiveCharacterTextSplitter | PDF 解析与文本分块 |
| 向量嵌入 | OpenAI Embeddings (via SiliconFlow) | 文本向量化 |
| 金融数据 | yfinance (Yahoo Finance API) | 股价历史数据获取 |
| 新闻检索 | Tavily Search API | 实时新闻搜索 |
| 可视化 | Matplotlib + Pandas | 股价走势图绘制 |
| 前端 | Streamlit | Web 交互界面 |
| 配置管理 | python-dotenv | 环境变量与密钥管理 |

---

## 五、保研简历撰写建议（简明版）

> **项目名称：** 基于 LLM Agent 的智能金融研报助手（Fin-Agent）
>
> **项目描述：** 设计并实现了一个基于大语言模型 Agent 的智能金融研报分析系统。采用 LangGraph 构建 ReAct 推理循环，Agent 自主调度股价查询（Yahoo Finance）、实时新闻检索（Tavily）等外部工具，结合 RAG（检索增强生成）技术对金融 PDF 报告进行语义检索，将多源异构金融数据融合后由 LLM 生成结构化深度分析报告。前端基于 Streamlit 构建交互式 Web 界面，实现了自然语言驱动的端到端金融分析流程。
>
> **关键技术：** LangGraph / LangChain、ReAct Agent、RAG（FAISS + Embedding）、Tool Calling、DeepSeek-V3、Streamlit
>
> **个人贡献/亮点（按需选取）：**
> - 设计了基于有向状态图的 Agent 推理架构，实现 LLM 对多工具的自主调度
> - 构建了 RAG 流水线（PDF 解析 → 文本分块 → 向量嵌入 → FAISS 索引），支持对长文档的语义检索
> - 实现了多级降级容错机制，保证系统在不同依赖版本下的兼容性
> - 融合股价时序数据、新闻文本、PDF 财报三类异构数据源，由 Agent 统一推理分析

---

## 六、涉及的核心学术概念

| 概念 | 关键论文/来源 |
|------|-------------|
| ReAct Agent | Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023 |
| RAG | Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020 |
| Tool Use / Function Calling | Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools", NeurIPS 2023 |
| 向量相似度搜索 (FAISS) | Johnson et al., "Billion-scale similarity search with GPUs", IEEE TBD 2021 |
| 文本嵌入 (Embedding) | Neelakantan et al., "Text and Code Embeddings by Contrastive Pre-Training", 2022 |
| LangGraph 状态图 | LangChain 官方文档, Multi-Agent Orchestration Framework |
