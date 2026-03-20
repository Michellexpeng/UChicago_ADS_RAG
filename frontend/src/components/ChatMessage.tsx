import { GraduationCap, User, ExternalLink } from 'lucide-react'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  sources?: string[]
}

function formatTime(d: Date) {
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
}

export default function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === 'user'

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
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap
            ${isUser
              ? 'rounded-tr-sm bg-[#800000] text-white'
              : 'rounded-tl-sm bg-white text-gray-800 shadow-sm border border-gray-100'
            }`}
        >
          {message.content}
        </div>

        {/* Source pills */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1.5">
            {message.sources.map(url => {
              // Show just the last path segment as label
              const label = url.split('/').filter(Boolean).pop() ?? url
              return (
                <a
                  key={url}
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 rounded-full border border-[#800000]/20
                    bg-[#800000]/5 px-2.5 py-0.5 text-xs text-[#800000]
                    hover:bg-[#800000]/10 transition-colors"
                >
                  <ExternalLink className="h-3 w-3" />
                  {label}
                </a>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
