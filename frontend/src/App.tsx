import { useState, useRef, useEffect } from 'react'
import { GraduationCap } from 'lucide-react'
import ChatMessage, { Message } from './components/ChatMessage'
import SampleQuestions from './components/SampleQuestions'
import ChatInput from './components/ChatInput'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        "Welcome to the UChicago Applied Data Science Q&A Assistant! I'm here to help answer your questions about the Applied Data Science master's program. Feel free to ask about admissions, curriculum, career outcomes, or browse the sample questions on the left.",
      timestamp: new Date(),
      sources: [],
      references: [],
    },
  ])
  const [isLoading, setIsLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleSend = async (content: string) => {
    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
      sources: [],
      references: [],
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    const assistantId = (Date.now() + 1).toString()

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: content }),
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
        }
      }
    } catch (err) {
      setIsLoading(false)
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
    <div className="flex h-screen overflow-hidden bg-white">
      {/* ── Sidebar ── */}
      <aside className="flex h-full w-80 shrink-0 flex-col border-r border-gray-200 bg-white">
        <SampleQuestions onQuestionClick={handleSend} />
      </aside>

      {/* ── Chat pane ── */}
      <main className="flex flex-1 flex-col overflow-hidden bg-gray-50">

        {/* Header */}
        <header className="shrink-0 bg-[#800000] px-5 py-4 text-white">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/10">
              <GraduationCap className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-[17px] font-semibold leading-tight">
                Applied Data Science Q&amp;A
              </h1>
              <p className="text-xs text-white/70">University of Chicago · Master's Program</p>
            </div>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
          {messages.map(msg => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          {/* Typing indicator — shown only before first token arrives */}
          {isLoading && (
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#800000]">
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

        {/* Input */}
        <div className="shrink-0 border-t border-gray-200 bg-white">
          <ChatInput onSendMessage={handleSend} disabled={isLoading} />
        </div>
      </main>
    </div>
  )
}
