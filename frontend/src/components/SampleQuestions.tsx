import { Lightbulb } from 'lucide-react'

const CATEGORIES = [
  {
    label: 'Program Information',
    questions: [
      'What is the duration of the Applied Data Science program?',
      'What are the admission requirements for the program?',
      'Is the program offered online or on-campus?',
      'What is the tuition cost for the Applied Data Science program?',
    ],
  },
  {
    label: 'Curriculum',
    questions: [
      'What courses are required in the Applied Data Science curriculum?',
      'Are there any specialization tracks available?',
      'What programming languages will I learn?',
      'Does the program include a capstone project?',
    ],
  },
  {
    label: 'Career & Outcomes',
    questions: [
      'What career opportunities are available after graduation?',
      'What is the average salary for program graduates?',
      'Does the program offer career support services?',
      'What companies hire UChicago ADS graduates?',
    ],
  },
  {
    label: 'Application Process',
    questions: [
      'When is the application deadline?',
      'Is work experience required for admission?',
      'Are GRE scores required?',
      'How do I apply for financial aid or scholarships?',
    ],
  },
]

interface Props {
  onQuestionClick: (q: string) => void
}

export default function SampleQuestions({ onQuestionClick }: Props) {
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="shrink-0 border-b border-gray-200 px-4 py-4">
        <div className="flex items-center gap-2">
          <Lightbulb className="h-4 w-4 text-[#800000]" />
          <h2 className="font-semibold text-gray-800">Sample Questions</h2>
        </div>
        <p className="mt-0.5 pl-6 text-xs text-gray-500">Click any question to ask</p>
      </div>

      {/* Scrollable list */}
      <div className="flex-1 overflow-y-auto py-2">
        {CATEGORIES.map(cat => (
          <div key={cat.label} className="mb-2">
            <p className="px-4 py-2 text-[10.5px] font-semibold uppercase tracking-widest text-[#800000]">
              {cat.label}
            </p>
            {cat.questions.map(q => (
              <button
                key={q}
                onClick={() => onQuestionClick(q)}
                className="w-full border-l-2 border-transparent px-4 py-2.5 text-left
                  text-[13px] text-gray-600 transition-all
                  hover:border-[#800000] hover:bg-[#fdf5f5] hover:text-[#800000]"
              >
                {q}
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
