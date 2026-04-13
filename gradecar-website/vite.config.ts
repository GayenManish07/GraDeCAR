export default defineConfig(({mode}) => {
  // ...
  return {
    base: '/GraDeCAR/', // Add this line!
    plugins: [react(), tailwindcss()],
    // ...
  };
});