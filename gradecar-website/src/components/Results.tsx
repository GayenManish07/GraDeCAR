import { motion } from 'motion/react';
import { TrendingUp, ShieldCheck, Database } from 'lucide-react';

export function Results() {
  return (
    <section id="results" className="py-20 bg-white">
      <div className="container mx-auto px-4 max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Key Findings</h2>
          <p className="text-xl text-[var(--color-ink)]/70 max-w-3xl mx-auto">
            Evaluated on APTOS 2019 and HAM10000 datasets under various noise conditions.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="flex flex-col items-center text-center p-6"
          >
            <div className="w-16 h-16 bg-green-50 text-green-600 rounded-full flex items-center justify-center mb-6">
              <TrendingUp className="w-8 h-8" />
            </div>
            <h3 className="text-xl font-bold mb-3">Up to 5% Improvement</h3>
            <p className="text-[var(--color-ink)]/70">
              Achieves an average improvement of up to 5% in macro-F1 over state-of-the-art noisy-label learning methods.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="flex flex-col items-center text-center p-6"
          >
            <div className="w-16 h-16 bg-purple-50 text-purple-600 rounded-full flex items-center justify-center mb-6">
              <ShieldCheck className="w-8 h-8" />
            </div>
            <h3 className="text-xl font-bold mb-3">Robust to Noise</h3>
            <p className="text-[var(--color-ink)]/70">
              Demonstrates strong robustness to both symmetric and structured label noise, even at high noise rates (up to 50%).
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex flex-col items-center text-center p-6"
          >
            <div className="w-16 h-16 bg-orange-50 text-orange-600 rounded-full flex items-center justify-center mb-6">
              <Database className="w-8 h-8" />
            </div>
            <h3 className="text-xl font-bold mb-3">Handles Imbalance</h3>
            <p className="text-[var(--color-ink)]/70">
              Effectively mitigates the effects of severe class imbalance inherent in real-world medical datasets.
            </p>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="bg-[var(--color-ink)] text-white rounded-3xl p-8 md:p-12 text-center"
        >
          <h3 className="text-2xl md:text-3xl font-bold mb-6">Ablation Studies Highlight</h3>
          <p className="text-lg text-white/80 max-w-4xl mx-auto leading-relaxed">
            Our ablation studies demonstrate that GraDeCAR's performance arises from the interplay of model diversity through dual-branch agreement, contrastive representations enhanced by mixup, iterative but conservative dataset expansion, and controlled training via annealing and final retraining.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
