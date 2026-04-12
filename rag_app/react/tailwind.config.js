/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
    theme: {
        extend: {
            colors: {
                ocean: {
                    950: '#030d17',
                    900: '#041527',
                    800: '#0a2530',
                }
            }
        }
    },
    plugins: [],
}