import { Link } from 'react-router-dom';
import { ShieldCheck, Zap, BrainCircuit } from 'lucide-react';
import { motion } from 'framer-motion';
import './LandingPage.css';

const LandingPage = () => {
  return (
    <motion.div 
      className="landing-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
    >
      <div className="hero-section">
        <motion.h1 
          className="hero-title"
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          AI-Powered Loan Approvals in Seconds
        </motion.h1>
        <p className="hero-subtitle">
          Experience the future of finance. Fast, secure, and powered by state-of-the-art Machine Learning.
        </p>
        <Link to="/apply" className="btn-run hero-cta">
          <span className="btn-glow"></span>
          <span className="btn-text">Check Eligibility Now</span>
        </Link>
      </div>

      <div className="features-section">
        <motion.div 
          className="feature-card glass-panel"
          whileHover={{ y: -10 }}
        >
          <Zap className="feature-icon highlight-cyan" size={40} />
          <h3>Lightning Fast</h3>
          <p>Get your approval probability calculated instantly as you adjust your application details.</p>
        </motion.div>
        
        <motion.div 
          className="feature-card glass-panel"
          whileHover={{ y: -10 }}
        >
          <BrainCircuit className="feature-icon highlight-orange" size={40} />
          <h3>ML-Driven</h3>
          <p>Our XGBoost and Scikit-learn models analyze dozens of risk factors to provide accurate insights.</p>
        </motion.div>

        <motion.div 
          className="feature-card glass-panel"
          whileHover={{ y: -10 }}
        >
          <ShieldCheck className="feature-icon highlight-purple" size={40} />
          <h3>Highly Secure</h3>
          <p>Your financial data is processed locally with zero persistence, ensuring absolute privacy.</p>
        </motion.div>
      </div>

      <div className="trust-section">
        <p>Powered by Advanced XGBoost Decision Trees</p>
      </div>
    </motion.div>
  );
};

export default LandingPage;
