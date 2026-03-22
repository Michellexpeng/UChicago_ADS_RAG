import ReactMarkdown from 'react-markdown'
import { GraduationCap, User } from 'lucide-react'

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
      className="rounded border border-[#800000]/20 bg-[#800000]/5 px-2 py-1 text-xs text-[#800000] hover:bg-[#800000]/10 hover:underline transition-colors"
    >
      {label}
    </a>
  )
}

export default function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  const refs = message.references ?? []

  return (
    <div className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full
          ${isUser ? 'bg-gray-200' : 'bg-[#800000]'}`}
      >
        {isUser
          ? <User className="h-4 w-4 text-gray-600" />
          : <GraduationCap className="h-4 w-4 text-white" />}
      </div>

      {/* Bubble */}
      <div className={`flex max-w-[75%] flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}>
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
              ? 'rounded-tr-sm bg-[#800000] text-white whitespace-pre-wrap'
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
      </div>
    </div>
  )
}
