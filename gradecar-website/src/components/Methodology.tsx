import { motion } from 'motion/react';
import { Filter, Layers, RefreshCw, CheckCircle2 } from 'lucide-react';

const steps = [
  {
    icon: Filter,
    title: "1. Identification of Pristine Samples",
    description: "Employs CleanLab within a K-fold cross-validation setup. A backbone model is trained for three epochs to capture easily learnable patterns while avoiding memorization. Samples whose predicted labels match their provided labels are marked as confident."
  },
  {
    icon: Layers,
    title: "2. Training Deep Learning Models",
    description: "Trains two models in parallel: a ResNet-based CNN classifier and a contrastive learning model. The CNN is trained on confident samples using cross-entropy loss, while the contrastive model is trained using mixup-augmented samples to improve generalization."
  },
  {
    icon: RefreshCw,
    title: "3. Iterative Relabeling and Annealing",
    description: "Both models generate predictions on non-confident samples. A sample is relabeled only when both models agree with high confidence. Newly relabeled samples are added to the pristine set. Training epochs are annealed by 10% after each round to reduce error propagation."
  },
  {
    icon: CheckCircle2,
    title: "4. Inference",
    description: "During inference, both models output softmax vectors over the classes. The final prediction is computed by averaging the two probability vectors element-wise, enhancing robustness under noisy conditions while maintaining interpretability."
  }
];

export function Methodology() {
  return (
    <section id="methodology" className="py-20 bg-[var(--color-muted)]">
      <div className="container mx-auto px-4 max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Methodology & Architecture</h2>
          <p className="text-xl text-[var(--color-ink)]/70 max-w-3xl mx-auto">
            GraDeCAR integrates three main components to learn reliably from noisy and imbalanced medical image datasets.
          </p>
        </motion.div>

        {/* --- ARCHITECTURE GRAPHICS SECTION --- */}
        <div className="mb-20 space-y-12">
          
          {/* Figure 1 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="bg-white p-4 md:p-8 rounded-3xl shadow-sm border border-[var(--color-border)]"
          >
            <h3 className="text-2xl font-bold mb-6 text-center">Overall Framework Architecture</h3>
            <div className="w-full aspect-[2/1] md:aspect-[2.5/1] bg-gray-50 rounded-2xl border-2 border-dashed border-gray-200 relative overflow-hidden flex items-center justify-center">
              
              {/* CHANGE THE SRC BELOW TO YOUR IMAGE NAME */}
              <img 
                src="arch_final.png" 
                alt="Figure 1: Overall schematic of GraDeCAR" 
                className="absolute inset-0 w-full h-full object-contain p-4" 
              />
              <span className="text-gray-400 text-sm -z-10">Place figure1.png in public folder</span>
              
            </div>
            <p className="text-center text-sm text-[var(--color-ink)]/60 mt-4 font-medium">Figure 1: Overall schematic of GraDeCAR</p>
          </motion.div>

          {/* Figures 2 & 3 */}
          <div className="grid md:grid-cols-2 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="bg-white p-4 md:p-8 rounded-3xl shadow-sm border border-[var(--color-border)]"
            >
              <h3 className="text-xl font-bold mb-4 text-center">Confident Sample Identification</h3>
              <div className="w-full aspect-video bg-gray-50 rounded-2xl border-2 border-dashed border-gray-200 relative overflow-hidden flex items-center justify-center">
                
                {/* CHANGE THE SRC BELOW TO YOUR IMAGE NAME */}
                <img 
                  src="full_arch_new.png" 
                  alt="Figure 2: Relabelling Stage" 
                  className="absolute inset-0 w-full h-full object-contain p-4" 
                />
                <span className="text-gray-400 text-sm -z-10">Place figure2.png in public folder</span>

              </div>
              <p className="text-center text-sm text-[var(--color-ink)]/60 mt-4 font-medium">Figure 2: CleanLab Initialization</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-white p-4 md:p-8 rounded-3xl shadow-sm border border-[var(--color-border)]"
            >
              <h3 className="text-xl font-bold mb-4 text-center">Training & Relabeling Mechanism</h3>
              <div className="w-full aspect-video bg-gray-50 rounded-2xl border-2 border-dashed border-gray-200 relative overflow-hidden flex items-center justify-center">
                
                {/* CHANGE THE SRC BELOW TO YOUR IMAGE NAME */}
                <img 
                  src="training_relabelling.png" 
                  alt="Figure 3: Relabeling Process" 
                  className="absolute inset-0 w-full h-full object-contain p-4" 
                />
                <span className="text-gray-400 text-sm -z-10">Place figure3.png in public folder</span>

              </div>
              <p className="text-center text-sm text-[var(--color-ink)]/60 mt-4 font-medium">Figure 3: Relabeling Process</p>
            </motion.div>
          </div>
        </div>
        {/* --- END ARCHITECTURE GRAPHICS SECTION --- */}

        <h3 className="text-2xl font-bold mb-8 text-center">Step-by-Step Process</h3>
        <div className="grid md:grid-cols-2 gap-8">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="bg-white p-8 rounded-2xl shadow-sm border border-[var(--color-border)] hover:shadow-md transition-shadow"
            >
              <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center mb-6">
                <step.icon className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold mb-3">{step.title}</h3>
              <p className="text-[var(--color-ink)]/70 leading-relaxed">
                {step.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
