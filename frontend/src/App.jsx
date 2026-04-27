import { Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import Navbar from './components/Navbar';
import LandingPage from './pages/LandingPage';
import ApplicationWizard from './pages/ApplicationWizard';
import ResultsPage from './pages/ResultsPage';
import './App.css';

function App() {
  const location = useLocation();

  const getBgClass = (path) => {
    if (path === '/') return 'bg-landing';
    if (path === '/apply') return 'bg-apply';
    if (path === '/result') return 'bg-result';
    return '';
  };

  return (
    <div className={`app-container ${getBgClass(location.pathname)}`}>
      <Navbar />
      <main className="main-content">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<LandingPage />} />
            <Route path="/apply" element={<ApplicationWizard />} />
            <Route path="/result" element={<ResultsPage />} />
          </Routes>
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
