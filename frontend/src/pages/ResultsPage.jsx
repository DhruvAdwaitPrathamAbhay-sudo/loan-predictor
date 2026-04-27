import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { RefreshCcw, CheckCircle, AlertOctagon } from 'lucide-react';
import './ResultsPage.css';

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, formData } = location.state || {};

  const [displayProb, setDisplayProb] = useState(0);

  useEffect(() => {
    if (!result) {
      navigate('/apply');
      return;
    }

    const target = result.approval_probability * 100;
    const duration = 2000;
    const startTime = performance.now();

    const animate = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      setDisplayProb(target * easeProgress);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    requestAnimationFrame(animate);
  }, [result, navigate]);

  if (!result) return null;

  const probability = displayProb / 100;
  const isApproved = result.approved;

  return (
    <motion.div 
      className="results-container"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="glass-panel results-card">
        <div className="status-header">
          {isApproved ? (
            <CheckCircle size={48} className="status-icon success" />
          ) : (
            <AlertOctagon size={48} className="status-icon error" />
          )}
          <h1 className={isApproved ? 'success-text' : 'error-text'}>
            {result.message}
          </h1>
        </div>

        <div className="gauge-wrapper large">
          <svg className="circular-gauge" viewBox="0 0 150 150">
            <defs>
              <linearGradient id="approvedGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#00e5ff" />
                <stop offset="100%" stopColor="#00ff88" />
              </linearGradient>
              <linearGradient id="deniedGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#ff9900" />
                <stop offset="100%" stopColor="#ff3366" />
              </linearGradient>
            </defs>
            <path 
              className="gauge-arc-bg" 
              d="M 30,120 A 65,65 0 1,1 120,120"
              fill="none" strokeWidth="8" strokeLinecap="round" 
            />
            <path 
              className={`gauge-arc-fill ${isApproved ? 'approved' : 'denied'}`} 
              d="M 30,120 A 65,65 0 1,1 120,120"
              fill="none" strokeWidth="8" strokeLinecap="round"
              strokeDasharray="306"
              strokeDashoffset={306 - (306 * probability)}
            />
            <circle
               cx={75 - 65 * Math.cos(Math.PI/4 + (probability * 1.5 * Math.PI))}
               cy={75 - 65 * Math.sin(Math.PI/4 + (probability * 1.5 * Math.PI))}
               r="6"
               className={`gauge-needle ${isApproved ? 'approved' : 'denied'}`}
            />
          </svg>
          <div className="gauge-content">
            <span className="gauge-value">{displayProb.toFixed(1)}<small>%</small></span>
            <span className="gauge-label">Probability</span>
          </div>
        </div>

        <div className="summary-section">
          <h3>Application Summary</h3>
          <div className="summary-grid">
            <div className="summary-item">
              <span className="label">Amount</span>
              <span className="value">${formData.loan_amnt.toLocaleString()}</span>
            </div>
            <div className="summary-item">
              <span className="label">Term</span>
              <span className="value">{formData.term} Months</span>
            </div>
            <div className="summary-item">
              <span className="label">Income</span>
              <span className="value">${formData.annual_inc.toLocaleString()}</span>
            </div>
            <div className="summary-item">
              <span className="label">FICO</span>
              <span className="value">{formData.fico_score}</span>
            </div>
          </div>
        </div>

        <button className="btn-secondary restart-btn" onClick={() => navigate('/apply')}>
          <RefreshCcw size={18} /> Start New Application
        </button>
      </div>
    </motion.div>
  );
};

export default ResultsPage;
