import { Github, FileText, ExternalLink } from 'lucide-react';

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-[var(--color-border)] bg-[var(--color-surface)]/80 backdrop-blur-md">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-[var(--color-ink)] text-[var(--color-paper)] flex items-center justify-center rounded-md font-serif font-bold text-lg">
            G
          </div>
          <span className="font-semibold text-lg tracking-tight">GraDeCAR</span>
        </div>
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-[var(--color-ink)]/70">
          <a href="#abstract" className="hover:text-[var(--color-ink)] transition-colors">Abstract</a>
          <a href="#methodology" className="hover:text-[var(--color-ink)] transition-colors">Methodology</a>
          <a href="#results" className="hover:text-[var(--color-ink)] transition-colors">Results</a>
        </nav>
        <div className="flex items-center gap-4">
          <a 
            href="https://github.com/GayenManish07/GraDeCAR" 
            target="_blank" 
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm font-medium bg-[var(--color-ink)] text-[var(--color-paper)] px-4 py-2 rounded-full hover:bg-[var(--color-ink)]/90 transition-colors"
          >
            <Github className="w-4 h-4" />
            <span>GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
}
