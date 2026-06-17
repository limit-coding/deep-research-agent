# Mini DeepResearch Agent

后端实习面试准备项目。目标不是”能跑的 demo”，而是每个设计决策都能讲清楚 why。

## 核心流程

```
query
  │
  ▼
[decompose]  拆成 ≤4 个独立子问题
  │
  ▼
[search] ◄────────────────┐  用 Tavily 查子问题 / 反思后的 follow-up 问题
  │                        │
  ▼                        │
[synthesize] 综合全部来源写报告草稿       │
  │                        │
  ▼                        │
[reflect] 自我批评：信息够吗？引用对得上吗？
  │                        │
  ├─ 不充分 且 iteration < MAX_RESEARCH_ITERATIONS ─┘
  │
  └─ 充分 / 达到上限
       │
       ▼
   [output] 整理最终报告 + 来源列表 + 反思记录
```

5 个节点对应 `deep_research_agent/nodes.py` 里的 5 个函数，图的装配在 `graph.py`。

## 目录结构

```
deep_research_agent/
  state.py   # ResearchState：节点间传递的共享状态
  llm.py     # LLM provider 切换（Anthropic/OpenAI）
  tools.py   # 搜索工具的唯一出口（Tavily）
  nodes.py   # 5 个节点的业务逻辑 + 2 个 structured-output schema
  graph.py   # 节点装配 + 反思循环的路由逻辑
main.py      # CLI 入口，按需接入 Langfuse tracing
tests/       # mock 掉 LLM/搜索后验证图的连线逻辑，不需要真实 key
reference/   # clone 的 langchain-ai/deep_research_from_scratch，只读，不提交进 git
```

## 怎么跑

```bash
uv sync --extra dev
cp .env.example .env   # 填入真实的 ANTHROPIC_API_KEY/OPENAI_API_KEY、TAVILY_API_KEY
uv run pytest -q       # 不需要任何 key，验证图的逻辑
uv run python main.py "LangGraph 和 LangChain 的关系是什么？"
```

## 关键设计决策（面试官问到怎么答）

**Q: 为什么用显式的 5 节点图，不用 LangGraph 的 ReAct 模式让 LLM 自己决定何时搜索？**

对照过 `langchain-ai/deep_research_from_scratch` 的 notebook 2：它是 `llm_call → tool_node → llm_call` 的真 ReAct 循环，模型自己决定调 `tavily_search` 还是 `think_tool`（一个不做任何事、纯粹强迫模型把反思写成结构化输出的“假工具”）。这个模式更灵活，但控制权完全在 LLM 手里——每轮搜几次、什么时候停，都不可预测。

我们要的是确定性的拆解→搜索→综合→反思流程：方便单独评测每一步（Langfuse trace 里按节点名对齐）、方便预估搜索调用次数上限（成本可控）、也方便后续给"引用可信度"单独打分。代价是更死板，遇到需要灵活调整策略的复杂问题不如 ReAct 灵活——这是有意识的取舍，不是没想到。

**Q: 为什么 `search_results` 用 `Annotated[list[SearchResult], operator.add]`？**

LangGraph 默认每个节点返回的状态字段会覆盖上一次的值。但反思循环会让 `search` 节点被进入第二次，如果直接覆盖，第一轮搜到的来源就丢了——而 `synthesize`/`output` 需要看到全部历史来源才能做完整的引用检查。`operator.add` 这个 reducer 让多次返回的列表自动拼接而不是替换。

**Q: 为什么 `decompose`/`reflect` 用 `with_structured_output` 而不是让模型输出文本再手写解析？**

手写 parse "模型输出的 JSON 字符串"很脆弱（模型可能加 markdown 代码块、可能漏字段）。`with_structured_output` 基于 tool-calling，让模型直接产出符合 Pydantic schema 的结构化对象，schema 本身就是接口文档，少一层解析失败的风险。

**Q: 反思循环怎么保证不会死循环？**

两层保护：业务层在 `reflect_node` 里维护 `iteration` 计数，`route_after_reflect` 检查 `iteration < MAX_RESEARCH_ITERATIONS`（默认 2，.env 可调）；框架层 LangGraph 本身也有 `recursion_limit`（默认 25）作为兜底。只信任 LLM 的自我判断（"我觉得信息还不够"）是不够的——这正是 Reflexion 类方法最容易踩的坑。

**Q: `web_search` 为什么自己包一层，不直接在 node 里调 `TavilyClient`？**

把搜索源换成别的（比如免 key 的 DuckDuckGo，或者以后要支持的 MCP 工具）只需要改 `tools.py` 这一个函数的实现和返回的字段名，`nodes.py` 完全不用动。同时这里也是唯一一处吞掉网络异常返回空列表的地方——一次搜索失败不该让整条图崩掉，而是交给 `reflect` 节点去判断"这个子问题信息不足，要不要重查"。

**Q: 为什么 LLM 默认选 Anthropic Claude Haiku，而不是 GPT-4o-mini？**

两个都支持，由 `LLM_PROVIDER` 环境变量切换，`llm.py` 是唯一改动点。先用便宜模型控制迭代成本，等流程跑顺、评测指标稳定了再考虑要不要换更大的模型。`temperature=0` 是为了同一道评测题多次跑结果可比，不是为了所谓"更准确"。

**Q: 反思节点为什么说借鉴了 Karpathy autoresearch？**

那个项目里每轮迭代后会有一个 keep/revert 的自我评估步骤决定要不要采纳这次改动，而不是无条件相信模型的最新输出。我们的 `reflect_node` 是同一个思路在"研究报告"场景下的应用：每轮综合后先自我批评，批评不通过才追加搜索重新综合。项目本身完全是 DeepResearch Agent，没有借用它的训练循环或其他部分。

## 跟参考仓库的对比笔记

读 + 实跑了 `langchain-ai/deep_research_from_scratch`（clone 在 `reference/`，只读，不提交）：

- notebook 1（scoping）：纯结构化输出做澄清 + 生成 research brief，用 `Command` 做条件路由，这个我们目前的骨架里没有"反问用户澄清"这一步——属于明确的范围裁剪，不是疏漏（CLAUDE.md 里没要求多轮对话澄清）。
- notebook 2（research agent）：ReAct 循环 + `think_tool` 强制结构化反思，跟我们 `reflect_node` 的思路殊途同归，只是它把反思塞进了循环内部的每一步，我们是综合之后单独一步。
- notebook 4/5（supervisor）：用 `asyncio.gather` 并行跑多个子 agent 分头研究不同子主题，再汇总——这是我们 `decompose` 出多个子问题后值得借鉴的下一步优化方向（目前 `search_node` 是顺序查询）。

## 当前限制 / 下一步

- [ ] 评测数据集（10-20 题 + 标准答案）存进 Langfuse Dataset，LLM-as-judge 打分准确性 + 引用是否真支持结论
- [ ] `search_node` 改成并发查询（参考 supervisor 模式的 `asyncio.gather`），现在是顺序循环
- [ ] 长网页内容摘要压缩（参考 `deep_research_from_scratch` 的 `summarize_webpage_content`），现在直接用 Tavily 返回的 snippet
- [ ] 引用可信度打分（参考 `tarun7r/deep-research-agent` 的思路），现在只检查"引用编号对不对得上"，没有单独评估来源本身的可信度
- [ ] LangGraph checkpointer，支持长任务断点续跑/人工介入
