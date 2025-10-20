import { defineConfig } from '@tanstack/router-generator';

export default defineConfig({
  routesDirectory: './src/routes',
  outputFile: './src/routeTree.gen.ts',
  watch: false
});
