# Mini DeepResearch Agent

后端工程师面试准备项目。目标不是”能跑的 demo”，而是每个设计决策都能讲清楚 why。

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
eval/        # 评测数据集 + LLM-as-judge 评测脚本（见下方"跑评测"）
tests/       # mock 掉 LLM/搜索后验证图的连线逻辑，不需要真实 key
reference/   # clone 的 langchain-ai/deep_research_from_scratch，只读，不提交进 git
```

## 用到的开源工具是什么

不熟悉这套生态的话，下面是个速查表，重点是"在这个项目里它具体管什么"，不是泛泛的官方介绍：

- **uv**：Rust 写的 Python 包/项目管理器，比 `pip install` + `venv` 快很多，现在社区迁移很快。这里用它替代 pip：`uv sync` 一条命令建虚拟环境 + 装好 `pyproject.toml` 里声明的全部依赖，`uv.lock` 锁版本号（类似前端的 `package-lock.json`，保证别人 clone 下来装的版本跟你完全一致）。
- **LangChain / langchain-core**：给"调用 LLM、消息格式、工具调用"定义了一套统一接口，换模型厂商不用改业务代码。`llm.py` 里的 `ChatAnthropic`/`ChatOpenAI`、`nodes.py` 里的 `HumanMessage`，都是它定义的抽象。
- **LangGraph**：LangChain 团队做的"状态图"编排框架，比单纯一条 prompt 链更适合做有分支、循环的 agent 流程。我们的反思重试循环就是靠它的 `add_conditional_edges` 实现的，`graph.py` 是直接用它的地方。
- **Pydantic**：数据校验库，写一个 class 就能描述"这个数据应该长什么样"。`nodes.py` 里的 `SubQuestions`/`Critique` 用它定义 schema，配合 LangChain 的 `with_structured_output`，强迫 LLM 输出符合这个结构的数据，而不是一段随意文本再自己写正则去解析。
- **Tavily**：专门为 LLM agent 设计的搜索 API——返回的是适合喂给模型的摘要和正文，不是给人看的搜索结果网页，区别于自己爬 Google 或者调免费但不稳定的 DuckDuckGo。`tools.py` 是唯一调它的地方。
- **Langfuse**：LLM 应用的可观测性 + 评测平台，开源、可自部署也有免费云版。这个项目用到它两个能力：tracing（`main.py` 里的 `CallbackHandler`，记录每次跑 agent 时每个节点/每次 LLM 调用的输入输出和耗时，方便排查"为什么这次结果不对"）和 Dataset/Evaluation（还没做的下一步，存评测题 + LLM-as-judge 自动打分）。同类工具里 LangSmith 是 LangChain 官方产品但闭源，选 Langfuse 是 CLAUDE.md 里定好的技术选型。
- **pytest**：Python 最主流的测试框架。`tests/test_graph.py` 里用的 `@patch` 其实来自标准库 `unittest.mock`，不是 pytest 自带的，pytest 只是负责发现和运行这些测试函数。
- **gh CLI**：GitHub 官方命令行工具。这次创建仓库、推送代码全程没碰浏览器，等价于网页上点 "New repository" 再 `git remote add` 再 `git push`，命令行一步到位。

## 怎么跑

### 1. 装依赖

```bash
cd /Users/limit/deep-research-agent
uv sync --extra dev
```

`uv sync` 会在项目目录下建一个 `.venv/`，把 `pyproject.toml` 里声明的依赖都装进去。之后有两种方式执行命令：

- 每次命令前加 `uv run`，比如 `uv run pytest`——不用管虚拟环境有没有激活，`uv` 自动用对的环境。
- 或者先 `source .venv/bin/activate` 激活一次，之后这个终端窗口里直接 `pytest`、`python main.py` 就行，不用每次加前缀。退出用 `deactivate`。两种等价，看个人习惯。

### 2. 配置 API key

```bash
cp .env.example .env
```

打开 `.env` 填入真实值，每个变量的作用：

| 变量 | 作用 | 去哪儿拿 |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` 或 `openai`，决定 `llm.py` 用哪家 | 不用申请，直接选 |
| `ANTHROPIC_API_KEY` | provider 选 anthropic 时必填 | https://console.anthropic.com |
| `OPENAI_API_KEY` | provider 选 openai 时必填 | https://platform.openai.com/account/api-keys |
| `TAVILY_API_KEY` | 搜索功能必填，不填 `search` 节点会一直返回空结果 | https://app.tavily.com（免费额度每月 1000 次） |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 选填，不填就是不接 tracing，`main.py` 里会自动跳过 | https://cloud.langfuse.com |
| `MAX_RESEARCH_ITERATIONS` | 反思循环最多重新搜索几轮，默认 2 | 直接改数字 |

`.env` 已经在 `.gitignore` 里，不会被提交。

### 3. 先跑测试，确认骨架没问题（不需要任何 key）

```bash
uv run pytest -q
```

这一步把 `get_llm`/`web_search` 都 mock 掉了，只验证图的节点连线和状态流转对不对，几秒钟出结果，跟有没有配 key、key 是否有效完全无关。如果这一步过不了，先别管 key 的事，是代码本身有问题。

### 4. 真实跑一次

```bash
uv run python main.py "LangGraph 和 LangChain 的关系是什么？"
```

会依次打印每个节点的执行（如果之后接了 LangSmith/终端日志），最后输出大致是这样的结构：

```
========================================

<综合报告正文，每条结论后面带 [1] [2] 这样的编号>

## 参考来源
[1] 标题 - 摘要 (https://...)
[2] 标题 - 摘要 (https://...)

## 反思记录（共 1 轮）
<reflect 节点给出的自我批评文字>
```

如果配了 Langfuse key，这次运行会在 https://cloud.langfuse.com 的项目里出现一条完整 trace，能展开看每个节点的输入输出和耗时。

### 故障排查

- 报 `tavily.errors.MissingAPIKeyError`：`.env` 里没填 `TAVILY_API_KEY`，或者命令没有从这个目录下执行（`.env` 只在 cwd 下会被自动读取）。
- 报 `openai.AuthenticationError` / `anthropic.AuthenticationError`：key 填错了或者是失效的占位 key，去对应平台后台确认。
- `final_report` 里"参考来源"是空的：说明 `web_search` 全部失败了（看 `tools.py` 里 `logger.warning` 的输出），通常是 key 无效或者网络问题，不会让程序崩，但报告质量会很差。
- 跑得很慢：每多一轮反思循环就要多跑一遍 search + synthesize + reflect，三次 LLM 调用，正常现象，调小 `MAX_RESEARCH_ITERATIONS` 能加快但报告质量可能下降。

## 跑评测

```bash
uv run python -m eval.run_eval
```

这一步需要 `TAVILY_API_KEY`、`LLM_PROVIDER` 对应的 LLM key，以及 `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`（评测结果要写进 Langfuse Dataset，这三类 key 缺一不可，跟单纯跑 `main.py` 不一样——那个 Langfuse key 是选填的）。

做了什么：

1. 把 `eval/dataset.py` 里 15 道后端/分布式系统方向的问题 + 标准答案要点，灌进 Langfuse 上名为 `deepresearch-eval-v1` 的 Dataset（重复跑不会产生重复数据，`create_dataset`/`create_dataset_item` 都按 name/id upsert）。
2. 对每道题真实跑一次完整的 Agent 图（`research_task`），拿到 `final_report` 和这一题用到的全部 `search_results`。
3. 两个 LLM-as-judge evaluator 各自打分（0~1）：
   - `accuracy_evaluator`：报告结论是否覆盖标准答案要点、有没有事实错误。
   - `citation_support_evaluator`：报告里 `[编号]` 引用的结论，是否真的能在对应来源原文里找到支持——这是只检查"编号对不对得上"做不到的，单独评估来源内容本身能不能撑住结论。
4. 跑完在终端打印每题的输入/输出/打分，同时整个实验在 Langfuse 后台（Datasets → `deepresearch-eval-v1` → Runs）可视化查看。

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

**Q: 为什么"准确性"和"引用是否支持结论"拆成两个 evaluator，而不是让 judge 一次性打一个综合分？**

这两件事经常脱节：报告可以"结论对但引用乱标"，也可以"引用都对得上但漏了关键信息"。拆开打分能在 Langfuse 的 Run 视图里分别看每一项的平均分，定位问题更精确——如果只看一个混合分，分数低了不知道是哪类问题。CLAUDE.md 里把这两项分开列也是这个原因。

**Q: 评测脚本里 judge 直接调 `get_llm()`，跟被评测的 agent 用同一个模型，这样不会自己给自己打高分吗？**

是一个已知的取舍，不是没意识到。更严谨的做法是用更强的模型（比如 Claude Sonnet）专门当 judge，跟被测的便宜模型分开，避免"用同一个模型的偏好评价自己"这种系统性偏差。现在保持简单是因为评测脚本本身的正确性（数据集结构、两个分数怎么定义、怎么跟 Langfuse Dataset 接起来）是这一步要先验证的事，换 judge 模型只是把 `get_llm()` 换成单独写死的 `ChatAnthropic(model="...")`，后续指标不稳定再做。

**Q: 为什么 `eval/run_eval.py` 给每个 dataset item 显式传 `id`（用 `q01`/`q02`...而不是让 Langfuse 自动生成）？**

`create_dataset_item` 文档写明"传自己的 id 可以用来去重"。评测脚本会被反复跑（改完 prompt 想看分数有没有提升），如果每次都让后端生成随机 id，15 道题跑 10 次就堆出 150 条重复数据；用固定 id 让这一步天然幂等，重复跑只是更新同一批条目，不污染 Dataset。

**Q: `pyproject.toml` 里为什么除了 `langchain-core` 还要加裸的 `langchain`？**

这是搭评测脚本时顺手发现的一个真实 bug：`langfuse.langchain.CallbackHandler`（`main.py` 接 tracing 用的那个）内部会 `import langchain` 去判断版本号选哪套兼容逻辑，但项目最初只声明了 `langchain-core`/`langchain-anthropic`/`langchain-openai`，没有 `langchain` 本身——本地没配 Langfuse key 时这行代码根本不会跑到，测试也测不出来，配了 key 才会在运行时炸 `ModuleNotFoundError`。补上 `langchain>=1.0` 依赖后验证过 import 和现有测试都正常。

## 跟参考仓库的对比笔记

读 + 实跑了 `langchain-ai/deep_research_from_scratch`（clone 在 `reference/`，只读，不提交）：

- notebook 1（scoping）：纯结构化输出做澄清 + 生成 research brief，用 `Command` 做条件路由，这个我们目前的骨架里没有"反问用户澄清"这一步——属于明确的范围裁剪，不是疏漏（CLAUDE.md 里没要求多轮对话澄清）。
- notebook 2（research agent）：ReAct 循环 + `think_tool` 强制结构化反思，跟我们 `reflect_node` 的思路殊途同归，只是它把反思塞进了循环内部的每一步，我们是综合之后单独一步。
- notebook 4/5（supervisor）：用 `asyncio.gather` 并行跑多个子 agent 分头研究不同子主题，再汇总——这是我们 `decompose` 出多个子问题后值得借鉴的下一步优化方向（目前 `search_node` 是顺序查询）。

## 当前限制 / 下一步

- [x] 评测数据集（15 题 + 标准答案）存进 Langfuse Dataset，LLM-as-judge 打分准确性 + 引用是否真支持结论（见上方"跑评测"；judge 跟被测 agent 用同一模型，是已知的简化，见 FAQ）
- [ ] `search_node` 改成并发查询（参考 supervisor 模式的 `asyncio.gather`），现在是顺序循环
- [ ] 长网页内容摘要压缩（参考 `deep_research_from_scratch` 的 `summarize_webpage_content`），现在直接用 Tavily 返回的 snippet
- [ ] 引用可信度打分（参考 `tarun7r/deep-research-agent` 的思路），现在只检查"引用编号对不对得上"，没有单独评估来源本身的可信度
- [ ] LangGraph checkpointer，支持长任务断点续跑/人工介入
