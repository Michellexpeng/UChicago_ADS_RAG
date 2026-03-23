import { BookOpen, Briefcase, ClipboardList, GraduationCap } from 'lucide-react'

const QUICK_QUESTIONS = [
  { icon: GraduationCap, label: 'Program Info', question: 'Tell me about the Applied Data Science program?' },
  { icon: BookOpen, label: 'Curriculum', question: 'What courses are provided in the curriculum?' },
  { icon: Briefcase, label: 'Careers', question: 'What career opportunities are available after graduation?' },
  { icon: ClipboardList, label: 'Application', question: 'What are the admission requirements?' },
]

interface Props {
  onQuestionClick: (q: string) => void
}

export default function QuickQuestions({ onQuestionClick }: Props) {
  return (
    <div className="grid grid-cols-2 gap-2 px-4 md:hidden">
      {QUICK_QUESTIONS.map(({ icon: Icon, label, question }) => (
        <button
          key={label}
          onClick={() => onQuestionClick(question)}
          className="flex flex-col items-start gap-2 rounded-xl border border-gray-200 bg-white p-3 text-left transition-colors duration-150 hover:border-primary/30 hover:bg-accent active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
        >
          <Icon className="h-4 w-4 text-primary" />
          <span className="text-xs font-medium text-gray-700">{label}</span>
        </button>
      ))}
    </div>
  )
}
