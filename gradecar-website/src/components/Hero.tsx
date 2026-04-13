import { motion } from 'motion/react';
import { FileText, Github } from 'lucide-react';

export function Hero() {
  return (
    <section className="pt-24 pb-16 md:pt-32 md:pb-24 px-4 overflow-hidden relative">
      {/* Background decorative elements */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-blue-50 rounded-full blur-3xl -z-10 opacity-50" />
      
      <div className="container mx-auto max-w-5xl text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 text-blue-700 text-sm font-medium mb-6 border border-blue-100">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
            </span>
            Medical Image Classification
          </div>
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight text-[var(--color-ink)] mb-6 leading-[1.1]">
            Gradual Denoising by <br className="hidden md:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
              Contrastive Agreement-based Relabeling
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-[var(--color-ink)]/70 max-w-3xl mx-auto mb-10 font-light">
            Tackling Label Noise in Medical Imaging through progressive cleaning, mixup augmentation, and dual-model agreement.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="flex flex-wrap justify-center gap-8 mb-12"
        >
          <div className="flex flex-col items-center">
            <span className="font-semibold text-lg">Manish Gayen</span>
            <span className="text-sm text-[var(--color-ink)]/60">IISER Bhopal, India</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="font-semibold text-lg">Samiran Das</span>
            <span className="text-sm text-[var(--color-ink)]/60">IISER Bhopal, India</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="font-semibold text-lg">Shubhobrata Bhattacharya</span>
            <span className="text-sm text-[var(--color-ink)]/60">IIT Kharagpur, India</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="flex flex-wrap justify-center gap-4"
        >
          <a
            href="https://github.com/GayenManish07/GraDeCAR"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-6 py-3 bg-[var(--color-ink)] text-white rounded-full font-medium hover:bg-[var(--color-ink)]/90 transition-all hover:scale-105 active:scale-95"
          >
            <Github className="w-5 h-5" />
            View on GitHub
          </a>
          <a
            href="#abstract"
            className="flex items-center gap-2 px-6 py-3 bg-white text-[var(--color-ink)] border border-[var(--color-border)] rounded-full font-medium hover:bg-gray-50 transition-all hover:scale-105 active:scale-95 shadow-sm"
          >
            <FileText className="w-5 h-5" />
            Read Abstract
          </a>
        </motion.div>
      </div>
    </section>
  );
}
