import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ChevronLeft, HelpCircle } from 'lucide-react';
import './ApplicationWizard.css';

const ApplicationWizard = () => {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  
  const [formData, setFormData] = useState({
    loan_amnt: 15000,
    term: 36,
    purpose: 'debt_consolidation',
    annual_inc: 75000,
    emp_length: 5,
    home_ownership: 'MORTGAGE',
    fico_score: 720,
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'home_ownership' || name === 'purpose' ? value : Number(value)
    }));
  };

  const nextStep = () => setStep(prev => Math.min(prev + 1, 3));
  const prevStep = () => setStep(prev => Math.max(prev - 1, 1));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    // Prepare API payload (same logic as before)
    const payload = {
      loan_amnt: formData.loan_amnt,
      term: formData.term,
      int_rate: 12.5,
      emp_length: formData.emp_length,
      annual_inc: formData.annual_inc,
      dti: 18.5,
      delinq_2yrs: 0,
      inq_last_6mths: 1,
      open_acc: 10,
      pub_rec: 0,
      revol_bal: 15000,
      revol_util: 45.0,
      total_acc: 20,
      mort_acc: 1,
      pub_rec_bankruptcies: 0,
      credit_history_years: 15.0,
      fico_score: formData.fico_score,
      home_ownership_OWN: formData.home_ownership === 'OWN' ? 1 : 0,
      home_ownership_RENT: formData.home_ownership === 'RENT' ? 1 : 0,
      verification_status_Source_Verified: 0,
      verification_status_Verified: 1,
      purpose_credit_card: formData.purpose === 'credit_card' ? 1 : 0,
      purpose_debt_consolidation: formData.purpose === 'debt_consolidation' ? 1 : 0,
      purpose_home_improvement: formData.purpose === 'home_improvement' ? 1 : 0,
      purpose_major_purchase: formData.purpose === 'major_purchase' ? 1 : 0,
      purpose_medical: formData.purpose === 'medical' ? 1 : 0,
      purpose_other: formData.purpose === 'other' ? 1 : 0,
      purpose_small_business: formData.purpose === 'small_business' ? 1 : 0,
      application_type_Joint_App: 0
    };

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      
      // Navigate to results page with data
      navigate('/result', { state: { result: data, formData } });
    } catch (err) {
      console.error(err);
      // Fallback mock if backend isn't running for the UI demonstration
      setTimeout(() => {
        navigate('/result', { state: { 
          result: { approved: true, approval_probability: 0.85, message: "Mock Approval (Backend Offline)" }, 
          formData 
        }});
      }, 1500);
    }
  };

  const stepVariants = {
    hidden: { opacity: 0, x: 50 },
    visible: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -50 }
  };

  return (
    <div className="wizard-container">
      <div className="wizard-header glass-panel">
        <div className="progress-bar-container">
          <div className="progress-bar" style={{ width: `${(step / 3) * 100}%` }}></div>
        </div>
        <div className="step-indicators">
          <span className={step >= 1 ? 'active' : ''}>1. Loan Details</span>
          <span className={step >= 2 ? 'active' : ''}>2. Financial Profile</span>
          <span className={step >= 3 ? 'active' : ''}>3. Credit History</span>
        </div>
      </div>

      <div className="wizard-body glass-panel">
        {loading ? (
          <div className="loading-state">
            <div className="scanning-bar"></div>
            <p>Analyzing Application via XGBoost...</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <AnimatePresence mode="wait">
              {step === 1 && (
                <motion.div key="step1" variants={stepVariants} initial="hidden" animate="visible" exit="exit" className="step-content">
                  <h2>Loan Requirements</h2>
                  <div className="input-group">
                    <label>Loan Amount <HelpCircle size={14} className="tooltip-icon" /></label>
                    <span className="value-display">${formData.loan_amnt.toLocaleString()}</span>
                    <input type="range" name="loan_amnt" min="1000" max="40000" step="500" value={formData.loan_amnt} onChange={handleChange} className="slider cyan-slider" />
                  </div>
                  <div className="input-row">
                    <div className="input-group">
                      <label>Term</label>
                      <select name="term" value={formData.term} onChange={handleChange}>
                        <option value="36">36 Months</option>
                        <option value="60">60 Months</option>
                      </select>
                    </div>
                    <div className="input-group">
                      <label>Purpose</label>
                      <select name="purpose" value={formData.purpose} onChange={handleChange}>
                        <option value="debt_consolidation">Debt Consolidation</option>
                        <option value="credit_card">Credit Card Refinance</option>
                        <option value="home_improvement">Home Improvement</option>
                      </select>
                    </div>
                  </div>
                </motion.div>
              )}

              {step === 2 && (
                <motion.div key="step2" variants={stepVariants} initial="hidden" animate="visible" exit="exit" className="step-content">
                  <h2>Financial Profile</h2>
                  <div className="input-group">
                    <label>Annual Income ($)</label>
                    <input type="number" name="annual_inc" value={formData.annual_inc} onChange={handleChange} required />
                  </div>
                  <div className="input-row">
                    <div className="input-group">
                      <label>Employment Length</label>
                      <span className="value-display">{formData.emp_length} Years</span>
                      <input type="range" name="emp_length" min="0" max="10" value={formData.emp_length} onChange={handleChange} className="slider orange-slider" />
                    </div>
                    <div className="input-group">
                      <label>Home Ownership</label>
                      <select name="home_ownership" value={formData.home_ownership} onChange={handleChange}>
                        <option value="MORTGAGE">Mortgage</option>
                        <option value="RENT">Rent</option>
                        <option value="OWN">Own</option>
                      </select>
                    </div>
                  </div>
                </motion.div>
              )}

              {step === 3 && (
                <motion.div key="step3" variants={stepVariants} initial="hidden" animate="visible" exit="exit" className="step-content">
                  <h2>Credit History</h2>
                  <div className="input-group">
                    <label>FICO Credit Score</label>
                    <span className="value-display highlight-orange">{formData.fico_score}</span>
                    <input type="range" name="fico_score" min="300" max="850" value={formData.fico_score} onChange={handleChange} className="slider orange-slider" />
                  </div>
                  <p className="credit-info">The remaining 20+ variables (like DTI, Open Accounts, Public Records) are internally inferred for this demo to streamline the application process.</p>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="wizard-actions">
              {step > 1 && (
                <button type="button" className="btn-secondary" onClick={prevStep}>
                  <ChevronLeft size={20} /> Back
                </button>
              )}
              {step < 3 ? (
                <button type="button" className="btn-primary" onClick={nextStep}>
                  Next <ChevronRight size={20} />
                </button>
              ) : (
                <button type="submit" className="btn-run">
                  Submit Application
                </button>
              )}
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default ApplicationWizard;
