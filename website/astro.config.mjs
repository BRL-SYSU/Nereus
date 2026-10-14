import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import tailwindcss from "@tailwindcss/vite";
export default defineConfig({
  site: "https://brl-sysu.github.io",
  base: "/Nereus",
  trailingSlash: "always",
  output: "static",
  integrations: [mdx()],
  vite: { plugins: [tailwindcss()] },
});
