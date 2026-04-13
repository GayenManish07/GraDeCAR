import { motion } from 'motion/react';

export function Abstract() {
  return (
    <section id="abstract" className="py-20 bg-white">
      <div className="container mx-auto px-4 max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-8 text-center">Abstract</h2>
          
          <div className="max-w-none text-[var(--color-ink)]/80 leading-relaxed text-lg">
            <p className="mb-6">
              Incorrect or noisy labels pose a major predicament in AI-based medical image classification, a high-stakes prediction task. Noisy labels not only hinder the learning process but also prompt the models to learn misleading features instead of prominent attributes.
            </p>
            <p className="mb-6">
              In this work, we propose <strong className="text-[var(--color-ink)] font-semibold">Gradual Denoising by Contrastive Agreement-based Relabeling (GraDeCAR)</strong>, a method designed to progressively clean noisy labels through several complementary strategies. GraDeCAR first identifies a clean subset of samples using early stopping, then trains models with different inductive biases under mixup augmentation to reduce overfitting, and finally uses agreement-based predictions for relabeling.
            </p>
            <p className="mb-6">
              The number of confident samples increases in each round, allowing the models to explore a wider and more reliable feature space. Although the conservative filtering approach utilized in this work limits the influence of noisy labels, contrastive model–guided relabeling steadily improves label quality and overall performance.
            </p>
            <p>
              Our experiments show that GraDeCAR achieves noticeable improvements even under class imbalance and high noise rates, demonstrating its potency. Across all evaluated noise settings and datasets, GraDeCAR yields an average improvement of up to <strong className="text-[var(--color-ink)] font-semibold">5% in macro-F1</strong> over state-of-the-art noisy-label learning methods.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
