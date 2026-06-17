# 项目目标

为后端工程师面试准备的项目：用 LangGraph 写一个 Mini DeepResearch Agent，
接 Langfuse 做全链路 tracing + 评测。目标不是"能跑的 demo"，而是能在面试里讲清楚每个设计决策。

# 范围（已确认，不要扩大）

只做一个项目：DeepResearch Agent。不做"Karpathy autoresearch 复刻"模式（曾讨论过把这个项目和
ML 实验自动调参的 dual-mode harness 合并，已放弃——那个需要真实 GPU 训练循环，时间/资源成本太高，
不在范围内）。不要重新提出做第二个模式。

# 核心流程

query 拆解 → 多轮 web search 工具调用（要有循环/重试，不是单轮问答）→ 信息综合 →
self-critique 反思步骤（Reflexion 式：发现信息不足或引用不可靠就重新检索）→ 输出报告

# 技术选型（已定，不要重新选型）

- Python 3.11+（Langfuse 集成要求）
- LangGraph 做状态图编排
- 搜索工具：Tavily 优先（有官方 LangChain Tool 封装）；先用免 key 的 DuckDuckGo 跑通流程也可以
- LLM：先用便宜模型（Claude Haiku / GPT-4o-mini）控制迭代成本，最后再考虑换大模型冲指标
- Langfuse 做 tracing（LangChain CallbackHandler 接入）+ Dataset/Evaluation 做评测，
  自部署或用 Cloud 免费层都行

# 评测

写一个 10-20 题的小数据集（问题 + 标准答案），存进 Langfuse Dataset，用 LLM-as-judge 自动打分：
- 准确性
- 引用是否真的支持给出的结论（这一项很多人懒得做，做了是加分点）

# 参考仓库（学习/对照用，不要照抄）

- https://github.com/langchain-ai/deep_research_from_scratch —— 教学向，先看这个理解模式
- https://github.com/langchain-ai/open_deep_research —— LangChain 官方生产级实现，用来对照差距
- https://github.com/tarun7r/deep-research-agent —— 引用可信度打分的实现思路可参考
- Langfuse 官方 LangGraph cookbook（同时演示 tracing 和 evaluation 怎么接）

# 注意事项

- 不要先搭一个"通用 Agent Harness 框架"再往里塞功能——直接做这一个项目，需要的抽象到时候自然会冒出来，
  不要为了"看起来像平台"过度设计。
- self-critique 反思步骤可以提一句"借鉴了 Karpathy autoresearch 的 keep/revert 迭代评估思路"，
  这是一个具体、站得住的引用，但项目本身不是在复刻它，别跑题去做那个。
- 不要参考 `/Users/limit/鱼皮资料/a鱼皮编程分享/AI资源/src`（以及同目录下任何疑似反编译/还原的
  Claude Code 源码）。来源不明、可能有 IP 问题，跟这个项目无关，不要读取或模仿其实现细节。
- 这是面试准备项目：代码质量之外，每个设计决策都要能讲清楚"为什么这样做"，不只是"用了什么技术"，
  写代码时多想一句"如果面试官问为什么这么设计，我答得上来吗"。

# 第一步

读完这份文件后，先跟用户确认一下范围理解是否一致，然后开始搭最小骨架：
LangGraph 图结构（节点：拆解 / 搜索 / 综合 / 反思 / 输出）+ Tavily 工具 + Langfuse tracing 接入。
