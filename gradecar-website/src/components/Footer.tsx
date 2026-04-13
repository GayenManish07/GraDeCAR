import { Github } from 'lucide-react';

export function Footer() {
  return (
    <footer className="bg-[var(--color-muted)] py-12 border-t border-[var(--color-border)]">
      <div className="container mx-auto px-4 text-center">
        <h2 className="text-2xl font-serif font-bold mb-6">GraDeCAR</h2>
        <p className="text-[var(--color-ink)]/60 max-w-2xl mx-auto mb-8">
          Gradual Denoising by Contrastive Agreement-based Relabeling for Tackling Label Noise in Medical Imaging
        </p>
        <div className="flex justify-center mb-8">
          <a
            href="https://github.com/GayenManish07/GraDeCAR"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-[var(--color-ink)]/70 hover:text-[var(--color-ink)] transition-colors"
          >
            <Github className="w-5 h-5" />
            <span>GitHub Repository</span>
          </a>
        </div>
        <p className="text-sm text-[var(--color-ink)]/40">
          © {new Date().getFullYear()} Manish Gayen, Samiran Das, Shubhobrata Bhattacharya.
        </p>
      </div>
    </footer>
  );
}
