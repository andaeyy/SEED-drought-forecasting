import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#172019",
        muted: "#5c685f",
        panel: "#ffffff",
        surface: "#edf2ed",
        field: "#f4f7f3",
        line: "#d9e1da",
        "line-strong": "#87958a",
        accent: "#397a48",
        "accent-strong": "#245f37",
        forest: "#173d2b",
        "forest-strong": "#0e2b1e",
        leaf: "#6da544",
        caution: "#9b6815",
        danger: "#a4362f",
        "drought-d0": "#FFFF00",
        "drought-d1": "#FCD37F",
        "drought-d2": "#FFAA00",
        "drought-d3": "#E60000",
        "drought-d4": "#730000"
      },
      boxShadow: {
        panel: "0 1px 2px rgba(23, 32, 25, 0.06), 0 8px 24px rgba(23, 61, 43, 0.06)",
        map: "0 1px 2px rgba(23, 32, 25, 0.08), 0 16px 36px rgba(23, 61, 43, 0.1)"
      }
    }
  },
  plugins: []
};

export default config;
