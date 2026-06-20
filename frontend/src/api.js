// 手写 fetch + ReadableStream 解析 SSE，而不是用浏览器原生 EventSource。
// 原因：EventSource 在收到非 200（比如限流的 429）时拿不到响应体/状态码，
// 只会触发一个不带信息的 error 事件，还会默认自动重连——对限流场景完全不可控。
// fetch 能先检查 response.ok，把 429 的 JSON detail 读出来再决定要不要重试。
export async function streamResearch(query, { onNode, onDone, signal }) {
  const response = await fetch(`/api/research?query=${encodeURIComponent(query)}`, { signal })

  if (!response.ok) {
    let detail = `请求失败（HTTP ${response.status}）`
    try {
      const body = await response.json()
      detail = body.detail || detail
    } catch {
      // 非 JSON 错误体，用上面的默认消息
    }
    throw new Error(detail)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)

      let eventName = null
      let data = null
      for (const line of rawEvent.split('\n')) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim()
        else if (line.startsWith('data:')) data = line.slice(5).trim()
      }
      if (!data) continue

      const payload = JSON.parse(data)
      if (eventName === 'node') onNode(payload)
      else if (eventName === 'done') onDone(payload)
    }
  }
}

export async function fetchExamples() {
  const response = await fetch('/api/examples')
  if (!response.ok) return []
  return response.json()
}
