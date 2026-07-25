/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: 'var(--background)',
        panel: 'var(--surface)',
        accent: 'var(--accent)',
        success: '#027A48',
        danger: '#D92D20',
      },
      boxShadow: {
        soft: 'var(--shadow-soft)',
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      borderRadius: {
        card: '14px',
      },
    },
  },
  plugins: [],
}
