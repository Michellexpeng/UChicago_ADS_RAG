import { useState, memo } from 'react'
import ReactMarkdown from 'react-markdown'
import { GraduationCap, User, Copy, Check, RefreshCw, RotateCw } from 'lucide-react'

export interface SourceReference {
  url: string
  title?: string
  snippet: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  sources?: string[]
  references?: SourceReference[]
}

function formatTime(d: Date) {
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
}

function SourceLink({ source }: { source: SourceReference }) {
  const label = source.title || source.url.split('/').filter(Boolean).pop() || source.url

  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-block max-w-[200px] truncate rounded-md border border-primary/20 bg-primary/5 px-2 py-1 text-xs text-primary transition-colors duration-150 hover:bg-primary/10 hover:underline"
    >
      {label}
    </a>
  )
}

const actionBtnClass = "flex items-center gap-1 rounded-md px-2 py-1 text-xs transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"

interface ChatMessageProps {
  message: Message
  isLastAssistant?: boolean
  onRegenerate?: () => void
  onRetry?: () => void
}

const ChatMessage = memo(function ChatMessage({ message, isLastAssistant, onRegenerate, onRetry }: ChatMessageProps) {
  const isUser = message.role === 'user'
  const refs = message.references ?? []
  const [copied, setCopied] = useState(false)
  const isError = !isUser && message.content.startsWith('⚠️')

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className={`group flex animate-message-in items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full
          ${isUser ? 'bg-gray-200' : 'bg-primary'}`}
      >
        {isUser
          ? <User className="h-4 w-4 text-gray-600" />
          : <GraduationCap className="h-4 w-4 text-white" />}
      </div>

      {/* Bubble */}
      <div className={`flex max-w-[90%] md:max-w-[75%] flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Name + time */}
        <div className={`flex items-center gap-2 text-xs text-gray-400 ${isUser ? 'flex-row-reverse' : ''}`}>
          <span className="font-medium text-gray-600">
            {isUser ? 'You' : 'UChicago ADS Assistant'}
          </span>
          <span>{formatTime(message.timestamp)}</span>
        </div>

        {/* Text */}
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed
            ${isUser
              ? 'rounded-tr-sm bg-primary text-white whitespace-pre-wrap'
              : 'rounded-tl-sm bg-white text-gray-800 shadow-sm border border-gray-100 prose prose-sm max-w-none'
            }`}
        >
          {isUser
            ? message.content
            : <ReactMarkdown>{message.content}</ReactMarkdown>
          }
        </div>

        {/* Source cards */}
        {!isUser && refs.length > 0 && (
          <div className="mt-1 flex flex-row flex-wrap gap-1.5">
            {[...new Map(refs.map(r => [r.title || r.url, r])).values()].map(r => (
              <SourceLink key={r.title || r.url} source={r} />
            ))}
          </div>
        )}

        {/* Action buttons */}
        {!isUser && message.content && (
          <div className="mt-0.5 flex gap-1 opacity-100 transition-opacity duration-150 md:opacity-0 md:group-hover:opacity-100">
            <button
              onClick={handleCopy}
              className={`${actionBtnClass} text-gray-400 hover:bg-gray-100 hover:text-gray-600`}
            >
              {copied
                ? <><Check className="h-3.5 w-3.5" /> Copied</>
                : <><Copy className="h-3.5 w-3.5" /> Copy</>}
            </button>
            {isLastAssistant && onRegenerate && !isError && (
              <button
                onClick={onRegenerate}
                className={`${actionBtnClass} text-gray-400 hover:bg-gray-100 hover:text-gray-600`}
              >
                <RefreshCw className="h-3.5 w-3.5" /> Regenerate
              </button>
            )}
            {isError && onRetry && (
              <button
                onClick={onRetry}
                className={`${actionBtnClass} text-primary hover:bg-accent`}
              >
                <RotateCw className="h-3.5 w-3.5" /> Retry
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
})

export default ChatMessage
