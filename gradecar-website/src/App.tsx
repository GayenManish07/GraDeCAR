import { Header } from './components/Header';
import { Hero } from './components/Hero';
import { Abstract } from './components/Abstract';
import { Methodology } from './components/Methodology';
import { Results } from './components/Results';
import { Footer } from './components/Footer';

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-grow">
        <Hero />
        <Abstract />
        <Methodology />
        <Results />
      </main>
      <Footer />
    </div>
  );
}
