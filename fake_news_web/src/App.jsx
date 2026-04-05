import { useState } from 'react';
import { Analytics } from '@vercel/analytics/react';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Falls back to localhost/IP for development, uses VITE_API_URL on Vercel
      const apiUrl = import.meta.env.VITE_API_URL || 'http://10.63.22.159:8000';
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to reach the AI server. Is the FastAPI backend running?');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="glass-container">
        <h1>📡 News Analyzer AI</h1>
        <p className="subtitle">Paste a news article below to instantly verify its authenticity using Machine Learning.</p>

      {error && (
        <div className="alert-warning">
          <strong>⚠️ Connection Error:</strong> {error}
        </div>
      )}

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste your news headline or full article text here..."
        disabled={loading}
      />

      <button 
        className="analyze-btn" 
        onClick={handleAnalyze} 
        disabled={loading || !text.trim()}
      >
        {loading ? (
          <>
            <span className="loading-spinner"></span>
            Analyzing Context...
          </>
        ) : (
          'Check Authenticity'
        )}
      </button>

      {result && (
        <div className={`result-card ${result.prediction === 1 ? 'result-real' : 'result-fake'}`}>
          <div className="result-title">
            {result.prediction === 1 ? '✅' : '🚫'} {result.label_string}
          </div>
          
          <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9em', marginTop: '5px' }}>
            Confidence interval: {result.prediction === 1 ? result.confidence.toFixed(1) : (100 - result.confidence).toFixed(1)}%
          </div>
          
          <div className="confidence-bar-bg">
            <div 
              className="confidence-bar-fill"
              style={{ width: `${result.prediction === 1 ? result.confidence : (100 - result.confidence)}%` }}
            ></div>
          </div>
        </div>
      )}
      </div>
      <Analytics />
    </>
  );
}

export default App;
