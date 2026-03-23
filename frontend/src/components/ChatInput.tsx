import { useState, useRef, KeyboardEvent, FormEvent } from 'react'
import { Send, Loader2 } from 'lucide-react'

interface Props {
  onSendMessage: (msg: string) => void
  disabled?: boolean
}

export default function ChatInput({ onSendMessage, disabled = false }: Props) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const submit = (e?: FormEvent) => {
    e?.preventDefault()
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSendMessage(trimmed)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <form onSubmit={submit} className="px-5 py-4">
      <div
        className={`flex items-end gap-2 rounded-xl border bg-white px-4 py-2.5 transition-colors duration-150
          ${disabled ? 'border-gray-200' : 'border-gray-300 focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/10'}`}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={e => { setValue(e.target.value); e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px' }}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question..."
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none bg-transparent text-base text-gray-800 outline-none
            placeholder:text-gray-400 disabled:opacity-50"
          style={{ minHeight: '24px', maxHeight: '120px' }}
        />
        <button
          type="submit"
          disabled={!value.trim() || disabled}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary
            text-white transition-colors duration-150 hover:bg-primary-hover active:scale-95
            disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-400
            focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
        >
          {disabled
            ? <Loader2 className="h-4 w-4 animate-spin" />
            : <Send className="h-4 w-4" />}
        </button>
      </div>
      <p className="mt-1.5 hidden text-center text-xs text-gray-400 md:block">
        Press Enter to send · Shift+Enter for new line
      </p>
    </form>
  )
}
