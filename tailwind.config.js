/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#F7F8FA',
        panel: '#FFFFFF',
        accent: '#101828',
        success: '#027A48',
        danger: '#D92D20',
      },
      boxShadow: {
        soft: '0 1px 0 rgba(16, 24, 40, 0.05)',
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
