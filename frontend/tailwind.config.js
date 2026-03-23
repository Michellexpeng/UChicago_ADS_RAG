/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#800000',
          foreground: '#ffffff',
          hover: '#a00000',
        },
        sidebar: {
          DEFAULT: '#ffffff',
          border: '#e5e7eb',
          foreground: '#111827',
          muted: '#6b7280',
          primary: '#800000',
        },
        border: '#e5e7eb',
        background: '#ffffff',
        foreground: '#111827',
        muted: { DEFAULT: '#f9fafb', foreground: '#6b7280' },
        accent: { DEFAULT: '#fdf5f5', foreground: '#111827' },
      },
      animation: {
        'message-in': 'message-in 0.25s ease-out',
        'dot-bounce': 'dot-bounce 1.3s ease-in-out infinite',
      },
      keyframes: {
        'message-in': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'dot-bounce': {
          '0%, 80%, 100%': { transform: 'translateY(0)', opacity: '0.4' },
          '40%': { transform: 'translateY(-5px)', opacity: '1' },
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
