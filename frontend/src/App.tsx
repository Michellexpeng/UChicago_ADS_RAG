import { useState, useRef, useEffect } from 'react'
import { GraduationCap, Menu, SquarePen, Square } from 'lucide-react'
import ChatMessage, { Message } from './components/ChatMessage'
import SampleQuestions from './components/SampleQuestions'
import ChatInput from './components/ChatInput'
import QuickQuestions from './components/QuickQuestions'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const welcomeMessage = (): Message[] => [{
  id: 'welcome',
  role: 'assistant',
  content:
    "Welcome to the UChicago Applied Data Science Q&A Assistant! I'm here to help answer your questions about the Applied Data Science master's program. Feel free to ask about admissions, curriculum, career outcomes, or browse the sample questions on the left.",
  timestamp: new Date(),
  sources: [],
  references: [],
}]

export default function App() {
  const [messages, setMessages] = useState<Message[]>(welcomeMessage())
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, isLoading])

  const handleNewChat = () => {
    abortRef.current?.abort()
    setMessages(welcomeMessage())
    setIsLoading(false)
    setIsStreaming(false)
  }

  const handleStop = () => {
    abortRef.current?.abort()
    setIsStreaming(false)
  }

  const handleRegenerate = () => {
    const lastUserMsg = [...messages].reverse().find(m => m.role === 'user')
    if (!lastUserMsg) return
    setMessages(prev => prev.filter(m => m.id !== prev[prev.length - 1].id))
    handleSend(lastUserMsg.content)
  }

  const handleRetry = () => {
    const lastUserMsg = [...messages].reverse().find(m => m.role === 'user')
    if (!lastUserMsg) return
    setMessages(prev => prev.filter(m => m.id !== prev[prev.length - 1].id))
    handleSend(lastUserMsg.content)
  }

  const handleSend = async (content: string) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
      sources: [],
      references: [],
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    const assistantId = crypto.randomUUID()

    try {
      abortRef.current = new AbortController()
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: content }),
        signal: abortRef.current.signal,
      })
      if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`)

      // Add empty assistant message — will be filled token by token
      setMessages(prev => [...prev, {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        sources: [],
        references: [],
      }])
      setIsLoading(false)
      setIsStreaming(true)

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const payload = line.slice(6).trim()
          if (payload === '[DONE]') break

          try {
            const event = JSON.parse(payload)
            if (event.type === 'token') {
              setMessages(prev => prev.map(msg =>
                msg.id === assistantId
                  ? { ...msg, content: msg.content + event.content }
                  : msg
              ))
            } else if (event.type === 'replace') {
              setMessages(prev => prev.map(msg =>
                msg.id === assistantId
                  ? { ...msg, content: event.content }
                  : msg
              ))
            } else if (event.type === 'sources') {
              setMessages(prev => prev.map(msg =>
                msg.id === assistantId
                  ? { ...msg, sources: event.sources, references: event.references }
                  : msg
              ))
            }
          } catch {
            // Skip malformed SSE data
          }
        }
      }
      setIsStreaming(false)
    } catch (err) {
      setIsLoading(false)
      setIsStreaming(false)
      if ((err as Error).name === 'AbortError') return
      setMessages(prev => [...prev, {
        id: assistantId,
        role: 'assistant',
        content: `⚠️ Couldn't reach the backend at \`${API_BASE}\`.\n\n${(err as Error).message}`,
        timestamp: new Date(),
        sources: [],
        references: [],
      }])
    }
  }

  return (
    <div className="flex h-full overflow-hidden bg-white">
      {/* ── Sidebar overlay (mobile) ── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/30 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Sidebar ── */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-72 flex-col border-r border-gray-200 bg-white transition-transform duration-200 md:static md:flex md:w-80 md:translate-x-0 ${
          sidebarOpen ? 'flex translate-x-0' : '-translate-x-full hidden md:flex'
        }`}
      >
        <SampleQuestions
          onQuestionClick={(q) => { handleSend(q); setSidebarOpen(false) }}
          onClose={() => setSidebarOpen(false)}
        />
      </aside>

      {/* ── Chat pane ── */}
      <main className="flex flex-1 flex-col overflow-hidden bg-gray-50">

        {/* Header */}
        <header className="shrink-0 bg-primary px-5 py-4 text-white">
          <div className="flex items-center gap-3">
            <button
              className="flex h-10 w-10 items-center justify-center rounded-full bg-white/10 transition-colors duration-150 hover:bg-white/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/30 md:hidden"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="hidden h-10 w-10 items-center justify-center rounded-full bg-white/10 md:flex">
              <GraduationCap className="h-5 w-5" />
            </div>
            <div className="flex-1">
              <h1 className="text-base font-semibold leading-tight">
                Applied Data Science Q&amp;A
              </h1>
              <p className="text-xs text-white/70">University of Chicago · Master's Program</p>
            </div>
            <button
              onClick={handleNewChat}
              className="flex h-9 w-9 items-center justify-center rounded-full bg-white/10 transition-colors duration-150 hover:bg-white/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/30"
              title="New chat"
            >
              <SquarePen className="h-4 w-4" />
            </button>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-5 py-5 space-y-5">
          {messages.map((msg, i) => {
            const lastAssistantIdx = messages.map(m => m.role).lastIndexOf('assistant')
            return (
              <ChatMessage
                key={msg.id}
                message={msg}
                isLastAssistant={i === lastAssistantIdx && !isLoading && !isStreaming}
                onRegenerate={i === lastAssistantIdx ? handleRegenerate : undefined}
                onRetry={i === lastAssistantIdx ? handleRetry : undefined}
              />
            )
          })}

          {/* Quick questions for mobile (empty chat) */}
          {messages.length === 1 && (
            <QuickQuestions onQuestionClick={handleSend} />
          )}

          {/* Typing indicator — shown only before first token arrives */}
          {isLoading && (
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary">
                <GraduationCap className="h-4 w-4 text-white" />
              </div>
              <div className="rounded-2xl rounded-tl-sm bg-white px-4 py-3 shadow-sm border border-gray-100">
                <div className="dot-bounce">
                  <span /><span /><span />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input / Stop button */}
        <div className="shrink-0 border-t border-gray-200 bg-white">
          {isStreaming ? (
            <div className="flex justify-center px-5 py-4">
              <button
                onClick={handleStop}
                className="flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2 text-sm text-gray-600 transition-colors duration-150 hover:bg-gray-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
              >
                <Square className="h-3.5 w-3.5 fill-current" /> Stop generating
              </button>
            </div>
          ) : (
            <ChatInput onSendMessage={handleSend} disabled={isLoading} />
          )}
        </div>
      </main>
    </div>
  )
}
