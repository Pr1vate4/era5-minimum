/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#0b1220',
        panel: '#111827',
        accent: '#60a5fa',
        success: '#22c55e',
        danger: '#ef4444',
      },
      boxShadow: {
        soft: '0 10px 30px rgba(15, 23, 42, 0.35)',
      },
      fontFamily: {
        sans: ['Inter', 'SF Pro Display', 'Segoe UI', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
