import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { fetchExamples, streamResearch, submitFeedback } from './api'
import './App.css'

const NODE_ORDER = ['decompose', 'search', 'synthesize', 'reflect', 'output']
const NODE_LABELS = {
  decompose: '拆解问题',
  search: '检索资料',
  synthesize: '综合报告',
  reflect: '自我反思',
  output: '整理输出',
}

function sourceDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

function SourceCard({ index, source }) {
  const domain = sourceDomain(source.url)
  return (
    <li className="source-card">
      <img
        className="source-favicon"
        src={`https://www.google.com/s2/favicons?sz=32&domain=${domain}`}
        alt=""
        onError={(event) => {
          event.currentTarget.style.visibility = 'hidden'
        }}
      />
      <div className="source-body">
        <div className="source-head">
          <span className="source-index">[{index + 1}]</span>
          <a className="source-title" href={source.url} target="_blank" rel="noreferrer">
            {source.title}
          </a>
        </div>
        <span className="source-domain">{domain}</span>
        <p className="source-snippet">{source.snippet}</p>
      </div>
    </li>
  )
}

function LiveSourceItem({ source }) {
  const domain = sourceDomain(source.url)
  return (
    <a className="live-source-item" href={source.url} target="_blank" rel="noreferrer">
      <img
        className="live-source-favicon"
        src={`https://www.google.com/s2/favicons?sz=16&domain=${domain}`}
        alt=""
        onError={(event) => {
          event.currentTarget.style.visibility = 'hidden'
        }}
      />
      <span className="live-source-title">{source.title}</span>
    </a>
  )
}

function ReflectionEntry({ reflection }) {
  return (
    <div className="reflection-entry">
      <div className="reflection-head">
        <span>第 {reflection.iteration} 轮反思</span>
        <span className={`verdict ${reflection.needs_more_research ? 'insufficient' : 'sufficient'}`}>
          {reflection.needs_more_research ? '信息不足，继续检索' : '信息充分'}
        </span>
      </div>
      <ReactMarkdown>{reflection.critique}</ReactMarkdown>
      {reflection.needs_more_research && reflection.follow_up_questions?.length > 0 && (
        <ul className="follow-up-list">
          {reflection.follow_up_questions.map((q) => (
            <li key={q}>{q}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

function App() {
  const [query, setQuery] = useState('')
  const [examples, setExamples] = useState([])
  const [status, setStatus] = useState('idle') // idle | streaming | done | error
  const [steps, setSteps] = useState([])
  const [searchRounds, setSearchRounds] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [feedback, setFeedback] = useState(null) // null | 'good' | 'bad'
  const abortRef = useRef(null)

  useEffect(() => {
    fetchExamples().then(setExamples)
  }, [])

  async function runResearch(question) {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setStatus('streaming')
    setSteps([])
    setSearchRounds(0)
    setResult(null)
    setError(null)
    setFeedback(null)

    let rounds = 0
    try {
      await streamResearch(question, {
        signal: controller.signal,
        onNode: (payload) => {
          if (payload.node === 'search') rounds += 1
          setSearchRounds(rounds)
          setSteps((prev) => [...prev, payload])
        },
        onDone: (payload) => {
          setResult(payload)
          setStatus('done')
        },
      })
    } catch (err) {
      if (err.name === 'AbortError') return
      setError(err.message)
      setStatus('error')
    }
  }

  function handleSubmit(event) {
    event.preventDefault()
    const question = query.trim()
    if (!question || status === 'streaming') return
    runResearch(question)
  }

  function runExample(question) {
    setQuery(question)
    runResearch(question)
  }

  const completedNodes = new Set(steps.map((step) => step.node))
  const isStreaming = status === 'streaming'

  const lastNode = steps.length > 0 ? steps[steps.length - 1].node : null
  // 还没收到任何事件时，正在跑的肯定是第一个节点；收到事件后，正在跑的大概率是
  // 顺序里的下一个（reflect 之后是回搜索还是去输出无法事先知道，这个近似已经够用——
  // 真实下一个事件到达时马上会被纠正，不影响最终结果，只是过程中偶尔有一帧猜错）。
  const currentNode = isStreaming
    ? lastNode === null
      ? NODE_ORDER[0]
      : NODE_ORDER[Math.min(NODE_ORDER.indexOf(lastNode) + 1, NODE_ORDER.length - 1)]
    : null

  const subQuestions = steps.find((step) => step.node === 'decompose')?.update.sub_questions ?? []
  const liveSources = steps
    .filter((step) => step.node === 'search')
    .flatMap((step) => step.update.search_results ?? [])
  const reflections = steps.filter((step) => step.node === 'reflect').map((step) => step.update)

  return (
    <div className="page">
      <header className="header">
        <span className="badge">LangGraph · Reflexion-style demo</span>
        <h1>Mini DeepResearch Agent</h1>
        <p className="subtitle">
          5 节点图：拆解 → 搜索 → 综合 → 反思（不充分则回到搜索） → 输出。
          {' '}
          <a href="https://github.com/limit-coding/deep-research-agent" target="_blank" rel="noreferrer">
            查看源码 →
          </a>
        </p>
      </header>

      <form className="query-form" onSubmit={handleSubmit}>
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="输入一个后端 / 分布式系统相关的问题…"
          rows={3}
          disabled={isStreaming}
        />
        <button type="submit" disabled={isStreaming}>
          {isStreaming ? '研究中…' : '开始研究'}
        </button>
      </form>

      {examples.length > 0 && (
        <div className="examples">
          <span>试试这些：</span>
          {examples.map((question) => (
            <button
              key={question}
              type="button"
              className="example-chip"
              onClick={() => runExample(question)}
              disabled={isStreaming}
            >
              {question}
            </button>
          ))}
        </div>
      )}

      {status !== 'idle' && (
        <div className="progress-panel">
          <ol className="progress">
            {NODE_ORDER.map((node) => {
              const state = completedNodes.has(node) && node !== currentNode
                ? 'done'
                : node === currentNode
                  ? 'active'
                  : 'pending'
              return (
                <li key={node} className={`step ${state}`}>
                  <span className="step-dot" />
                  {NODE_LABELS[node]}
                  {node === 'search' && searchRounds > 1 && <span className="step-badge">×{searchRounds}</span>}
                </li>
              )
            })}
          </ol>

          {subQuestions.length > 0 && (
            <ul className="sub-questions">
              {subQuestions.map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ul>
          )}

          {liveSources.length > 0 && (
            <div className="live-sources">
              <h3>检索过程（{liveSources.length} 条）</h3>
              <div className="live-source-list">
                {liveSources.map((source, index) => (
                  <LiveSourceItem key={`${source.url}-${index}`} source={source} />
                ))}
              </div>
            </div>
          )}

          {reflections.length > 0 && (
            <div className="live-reflections">
              {reflections.map((reflection) => (
                <ReflectionEntry key={reflection.iteration} reflection={reflection} />
              ))}
            </div>
          )}
        </div>
      )}

      {error && <p className="error">出错了：{error}</p>}

      {result && (
        <article className="report">
          <div className="stats-bar">
            <span>🔎 {subQuestions.length} 个子问题</span>
            <span>📚 {result.sources.length} 条来源</span>
            <span>♻️ 反思 {result.iteration} 轮</span>
            <span className="feedback-row">
              {feedback ? (
                <span className="feedback-thanks">已记录，感谢反馈</span>
              ) : (
                <>
                  <span className="feedback-label">报告质量：</span>
                  <button
                    type="button"
                    className="feedback-btn"
                    onClick={() => {
                      setFeedback('good')
                      submitFeedback(query, result.report, result.sources, 'good')
                    }}
                  >👍</button>
                  <button
                    type="button"
                    className="feedback-btn"
                    onClick={() => {
                      setFeedback('bad')
                      submitFeedback(query, result.report, result.sources, 'bad')
                    }}
                  >👎</button>
                </>
              )}
            </span>
          </div>

          <ReactMarkdown>{result.report}</ReactMarkdown>

          {result.sources.length > 0 && (
            <section className="sources">
              <h2>参考来源</h2>
              <ol className="source-list">
                {result.sources.map((source, index) => (
                  <SourceCard key={`${source.url}-${index}`} index={index} source={source} />
                ))}
              </ol>
            </section>
          )}

          {result.critique && (
            <section className="critique">
              <h2>反思记录（共 {result.iteration} 轮）</h2>
              <ReactMarkdown>{result.critique}</ReactMarkdown>
            </section>
          )}
        </article>
      )}

      <footer className="footer">
        <p>
          后端面试准备项目：代码、评测脚本、部署配置都在{' '}
          <a href="https://github.com/limit-coding/deep-research-agent" target="_blank" rel="noreferrer">
            GitHub repo
          </a>{' '}
          里。这是一个限流的公开 demo，单 IP 和每日总额度都有上限。
        </p>
      </footer>
    </div>
  )
}

export default App
