import { useState } from 'react';

const MAX_RETRIES = 3;
const RETRY_DELAYS = [2000, 6000, 10000]; // ms to wait before each retry
const FETCH_TIMEOUTS = [10000, 20000, 30000]; // ms timeout per attempt

const STATUS_MESSAGES = [
  '🔍 Analyzing your text...',
  '⏳ Waking up AI server — free tier takes a moment...',
  '🚀 Server is booting up, almost there...',
];

function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() =>
    clearTimeout(timer)
  );
}

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [statusMsg, setStatusMsg] = useState('');
  const [attempt, setAttempt] = useState(0);

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setStatusMsg(STATUS_MESSAGES[0]);
    setAttempt(0);

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    const payload = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    };

    let lastError = null;

    for (let i = 0; i < MAX_RETRIES; i++) {
      setAttempt(i + 1);
      setStatusMsg(STATUS_MESSAGES[Math.min(i, STATUS_MESSAGES.length - 1)]);

      try {
        const response = await fetchWithTimeout(
          `${apiUrl}/predict`,
          payload,
          FETCH_TIMEOUTS[i]
        );

        if (!response.ok) {
          const errBody = await response.text().catch(() => '');
          throw new Error(
            `Server returned ${response.status}${errBody ? ': ' + errBody : ''}`
          );
        }

        const data = await response.json();
        setResult(data);
        setLoading(false);
        setStatusMsg('');
        return; // success — exit
      } catch (err) {
        lastError = err;

        // Don't retry if it's an actual server error (4xx/5xx), only on network / timeout
        if (err.name !== 'AbortError' && err.message && !err.message.includes('Failed to fetch') && !err.message.includes('NetworkError') && !err.message.includes('Load failed')) {
          break;
        }

        // Wait before next retry (except last attempt)
        if (i < MAX_RETRIES - 1) {
          setStatusMsg(`${STATUS_MESSAGES[Math.min(i + 1, STATUS_MESSAGES.length - 1)]}`);
          await new Promise((r) => setTimeout(r, RETRY_DELAYS[i]));
        }
      }
    }

    // All retries exhausted
    setError(
      lastError?.name === 'AbortError'
        ? 'The AI server is still starting up. Please wait 30 seconds and try again.'
        : lastError?.message || 'Failed to connect to the server.'
    );
    setLoading(false);
    setStatusMsg('');
  };

  return (
    <div className="glass-container">
      <h1>📡 Fake News Detector</h1>
      <p className="subtitle">
        Paste a news article below to instantly verify its authenticity using
        Machine Learning.
      </p>

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
            {statusMsg}
          </>
        ) : (
          'Check Authenticity'
        )}
      </button>

      {loading && attempt > 1 && (
        <div className="retry-banner">
          <div className="retry-dots">
            <span className="retry-dot" />
            <span className="retry-dot" />
            <span className="retry-dot" />
          </div>
          <span>Attempt {attempt} of {MAX_RETRIES} — the free server needs a moment to wake up</span>
        </div>
      )}

      {result && (
        <div
          className={`result-card ${result.prediction === 1 ? 'result-real' : 'result-fake'}`}
        >
          <div className="result-title">
            {result.prediction === 1 ? '✅' : '🚫'} {result.label_string}
          </div>

          <div
            style={{
              color: 'rgba(255,255,255,0.7)',
              fontSize: '0.9em',
              marginTop: '5px',
            }}
          >
            Confidence interval:{' '}
            {result.prediction === 1
              ? result.confidence.toFixed(1)
              : (100 - result.confidence).toFixed(1)}
            %
          </div>

          <div className="confidence-bar-bg">
            <div
              className="confidence-bar-fill"
              style={{
                width: `${result.prediction === 1 ? result.confidence : 100 - result.confidence}%`,
              }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
