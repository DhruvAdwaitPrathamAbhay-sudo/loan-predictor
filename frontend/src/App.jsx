import { useState } from 'react'

function App() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  
  const [formData, setFormData] = useState({
    loan_amnt: 15000,
    annual_inc: 75000,
    term: 36,
    fico_score: 720,
    emp_length: 5,
    home_ownership: 'MORTGAGE',
    purpose: 'debt_consolidation'
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: name === 'home_ownership' || name === 'purpose' ? value : Number(value)
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    
    // Map minimal form to full API schema
    const payload = {
      loan_amnt: formData.loan_amnt,
      term: formData.term,
      int_rate: 12.5, // Default average
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
      
      // Categorical Mapping
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
    }

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      
      const data = await response.json()
      setResult(data)
    } catch (err) {
      console.error(err)
      alert("Error connecting to API. Is the server running?")
    } finally {
      setLoading(false)
    }
  }

  // Calculate SVG Gauge properties
  const radius = 100
  const circumference = 2 * Math.PI * radius
  const probability = result ? result.approval_probability : 0
  const offset = circumference - (probability * circumference)

  return (
    <div className="app-container">
      <header>
        <h1>Smart Finance Advisor</h1>
        <p>AI-Powered Loan Approval Prediction System</p>
      </header>

      <div className="dashboard-grid">
        {/* Left Column: Form */}
        <div className="glass-panel">
          <form onSubmit={handleSubmit} className="form-grid">
            
            <div className="form-group full-width">
              <label>Requested Loan Amount: ${formData.loan_amnt.toLocaleString()}</label>
              <input 
                type="range" 
                name="loan_amnt" 
                min="1000" max="40000" step="500" 
                value={formData.loan_amnt} 
                onChange={handleChange} 
              />
              <div className="slider-labels">
                <span>$1k</span>
                <span>$40k</span>
              </div>
            </div>

            <div className="form-group">
              <label>Annual Income ($)</label>
              <input 
                type="number" 
                name="annual_inc" 
                value={formData.annual_inc} 
                onChange={handleChange} 
                required 
              />
            </div>

            <div className="form-group">
              <label>FICO Credit Score</label>
              <input 
                type="number" 
                name="fico_score" 
                min="300" max="850" 
                value={formData.fico_score} 
                onChange={handleChange} 
                required 
              />
            </div>

            <div className="form-group">
              <label>Loan Term</label>
              <select name="term" value={formData.term} onChange={handleChange}>
                <option value="36">36 Months</option>
                <option value="60">60 Months</option>
              </select>
            </div>

            <div className="form-group">
              <label>Employment Length ({formData.emp_length} yrs)</label>
              <input 
                type="range" 
                name="emp_length" 
                min="0" max="10" 
                value={formData.emp_length} 
                onChange={handleChange} 
              />
            </div>

            <div className="form-group">
              <label>Home Ownership</label>
              <select name="home_ownership" value={formData.home_ownership} onChange={handleChange}>
                <option value="MORTGAGE">Mortgage</option>
                <option value="RENT">Rent</option>
                <option value="OWN">Own</option>
              </select>
            </div>

            <div className="form-group">
              <label>Loan Purpose</label>
              <select name="purpose" value={formData.purpose} onChange={handleChange}>
                <option value="debt_consolidation">Debt Consolidation</option>
                <option value="credit_card">Credit Card Refinance</option>
                <option value="home_improvement">Home Improvement</option>
                <option value="small_business">Small Business</option>
                <option value="major_purchase">Major Purchase</option>
              </select>
            </div>

            <div className="form-group full-width">
              <button type="submit" className="btn-submit" disabled={loading}>
                {loading ? 'Analyzing Application...' : 'Run Prediction Model'}
              </button>
            </div>

          </form>
        </div>

        {/* Right Column: Results */}
        <div className="glass-panel results-panel">
          {!result ? (
            <div className="placeholder-text">
              Enter applicant details and run the model to view XGBoost AI prediction.
            </div>
          ) : (
            <>
              <div className="gauge-container">
                <svg className="gauge-svg" viewBox="0 0 250 250">
                  <circle 
                    className="gauge-bg" 
                    cx="125" cy="125" r={radius} 
                  />
                  <circle 
                    className={`gauge-fill ${result.approved ? 'approved' : 'denied'}`} 
                    cx="125" cy="125" r={radius} 
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                  />
                </svg>
                <div className="gauge-center">
                  <div className="gauge-percentage">
                    {(probability * 100).toFixed(0)}%
                  </div>
                  <div className="gauge-label">Probability</div>
                </div>
              </div>

              <div className={`status-badge ${result.approved ? 'approved' : 'denied'}`}>
                {result.message}
              </div>

              <div className="analysis-list">
                <h3>Key Risk Factors Analyzed</h3>
                <div className="analysis-item">
                  <span>Loan-to-Income Ratio</span>
                  <span className="analysis-value">
                    {((formData.loan_amnt / formData.annual_inc) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="analysis-item">
                  <span>Credit Profile</span>
                  <span className="analysis-value">{formData.fico_score} FICO</span>
                </div>
                <div className="analysis-item">
                  <span>Algorithm</span>
                  <span className="analysis-value">XGBoost (Decision Trees)</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
