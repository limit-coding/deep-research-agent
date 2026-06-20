import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { fetchExamples, streamResearch } from './api'
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
  return (
    <li className="source-card">
      <span className="source-index">[{index + 1}]</span>
      <div className="source-body">
        <a className="source-title" href={source.url} target="_blank" rel="noreferrer">
          {source.title}
        </a>
        <span className="source-domain">{sourceDomain(source.url)}</span>
        <p className="source-snippet">{source.snippet}</p>
      </div>
    </li>
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

  return (
    <div className="page">
      <header className="header">
        <h1>Mini DeepResearch Agent</h1>
        <p className="subtitle">
          LangGraph 5 节点图：拆解 → 搜索 → 综合 → 反思（不充分则回到搜索） → 输出。
          {' '}
          <a href="https://github.com/limit-coding/deep-research-agent" target="_blank" rel="noreferrer">
            查看源码
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
        <ol className="progress">
          {NODE_ORDER.map((node) => (
            <li key={node} className={completedNodes.has(node) ? 'step done' : 'step'}>
              {NODE_LABELS[node]}
            </li>
          ))}
          {searchRounds > 1 && <li className="round-note">反思触发了第 {searchRounds} 轮检索</li>}
        </ol>
      )}

      {error && <p className="error">出错了：{error}</p>}

      {result && (
        <article className="report">
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
