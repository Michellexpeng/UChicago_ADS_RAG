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
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
