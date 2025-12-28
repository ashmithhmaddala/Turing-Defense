import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Code, Copy, Check, Terminal, Zap, Shield, Eye, ArrowRight } from '../components/Icons';

export default function SDKPage() {
  const [copiedIndex, setCopiedIndex] = useState(null);

  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const installCode = `<script src="https://your-domain.com/turing-defense.js"></script>`;
  
  const basicUsage = `// Initialize the detector
const detector = new TuringDefense({
  endpoint: 'https://your-api.com/analyze',
  sampleRate: 100,
  debug: false
});

// Start monitoring
detector.start();

// Get results
detector.onVerdict((result) => {
  console.log('Human probability:', result.score);
  console.log('Verdict:', result.verdict);
});`;

  const advancedConfig = `const detector = new TuringDefense({
  // API Configuration
  endpoint: 'https://your-api.com/analyze',
  apiKey: 'your-api-key',
  
  // Sampling
  sampleRate: 100,        // Events per second
  batchSize: 50,          // Events per API call
  batchInterval: 500,     // MS between batches
  
  // Features to track
  trackMouse: true,
  trackKeyboard: true,
  trackClicks: true,
  trackScroll: true,
  
  // Callbacks
  onVerdict: (result) => handleVerdict(result),
  onError: (error) => console.error(error),
  
  // Debug mode
  debug: process.env.NODE_ENV === 'development'
});`;

  const reactIntegration = `import { useEffect, useState } from 'react';
import TuringDefense from 'turing-defense';

function useBot Detection() {
  const [verdict, setVerdict] = useState(null);
  const [isHuman, setIsHuman] = useState(null);

  useEffect(() => {
    const detector = new TuringDefense({
      endpoint: '/api/analyze',
      onVerdict: (result) => {
        setVerdict(result);
        setIsHuman(result.verdict === 'human');
      }
    });

    detector.start();

    return () => detector.stop();
  }, []);

  return { verdict, isHuman };
}

// Usage in component
function ProtectedForm() {
  const { isHuman, verdict } = useBotDetection();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isHuman === false) {
      alert('Bot detected! Submission blocked.');
      return;
    }
    // Process form...
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Your form fields */}
      <button type="submit" disabled={isHuman === false}>
        Submit
      </button>
    </form>
  );
}`;

  const serverVerification = `// Express.js middleware example
const express = require('express');
const app = express();

app.post('/api/analyze', express.json(), async (req, res) => {
  const { events, sessionId } = req.body;
  
  // Forward to your Turing Defense backend
  const response = await fetch('http://localhost:5000/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ events })
  });
  
  const result = await response.json();
  
  // Store verdict for this session
  sessions.set(sessionId, result);
  
  res.json(result);
});

// Protected endpoint
app.post('/api/submit-form', (req, res) => {
  const sessionVerdict = sessions.get(req.body.sessionId);
  
  if (!sessionVerdict || sessionVerdict.verdict === 'bot') {
    return res.status(403).json({ error: 'Bot detected' });
  }
  
  // Process legitimate request...
});`;

  const codeBlocks = [
    { title: 'Installation', lang: 'html', code: installCode },
    { title: 'Basic Usage', lang: 'javascript', code: basicUsage },
    { title: 'Advanced Configuration', lang: 'javascript', code: advancedConfig },
    { title: 'React Integration', lang: 'jsx', code: reactIntegration },
    { title: 'Server-Side Verification', lang: 'javascript', code: serverVerification }
  ];

  return (
    <div className="sdk-page">
      {/* Hero */}
      <section className="sdk-hero">
        <div className="sdk-hero-content">
          <div className="sdk-badge">
            <Code size={16} />
            <span>JavaScript SDK</span>
          </div>
          <h1>Integrate Bot Detection in Minutes</h1>
          <p>
            Drop-in JavaScript SDK that invisibly monitors user behavior and 
            detects bots without any user friction. No CAPTCHAs, no puzzles.
          </p>
          <div className="sdk-hero-actions">
            <a href="#installation" className="btn btn-primary btn-lg">
              Get Started
            </a>
            <Link to="/demo" className="btn btn-secondary btn-lg">
              See It In Action
            </Link>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="sdk-features">
        <div className="sdk-container">
          <div className="sdk-features-grid">
            <div className="sdk-feature">
              <div className="sdk-feature-icon">
                <Zap size={24} />
              </div>
              <h3>Lightweight</h3>
              <p>Under 10KB gzipped. No dependencies. Won't slow down your site.</p>
            </div>
            <div className="sdk-feature">
              <div className="sdk-feature-icon">
                <Eye size={24} />
              </div>
              <h3>Invisible</h3>
              <p>Runs silently in the background. Users never know it's there.</p>
            </div>
            <div className="sdk-feature">
              <div className="sdk-feature-icon">
                <Shield size={24} />
              </div>
              <h3>Privacy-First</h3>
              <p>No PII collected. Only behavioral patterns are analyzed.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Code Examples */}
      <section className="sdk-code-section" id="installation">
        <div className="sdk-container">
          <h2>Integration Guide</h2>
          <p className="sdk-section-desc">
            Follow these steps to add Turing Defense to your website or application.
          </p>

          {codeBlocks.map((block, index) => (
            <div key={index} className="sdk-code-block" id={block.title.toLowerCase().replace(/\s+/g, '-')}>
              <div className="sdk-code-header">
                <div className="sdk-code-title">
                  <span className="sdk-step-number">{index + 1}</span>
                  <h3>{block.title}</h3>
                </div>
                <button 
                  className="copy-btn"
                  onClick={() => copyToClipboard(block.code, index)}
                >
                  {copiedIndex === index ? <Check size={14} /> : <Copy size={14} />}
                  <span>{copiedIndex === index ? 'Copied!' : 'Copy'}</span>
                </button>
              </div>
              <pre className="sdk-code-content">
                <code>{block.code}</code>
              </pre>
            </div>
          ))}
        </div>
      </section>

      {/* Configuration Reference */}
      <section className="sdk-config-section">
        <div className="sdk-container">
          <h2>Configuration Options</h2>
          <div className="sdk-table-wrapper">
            <table className="sdk-table">
              <thead>
                <tr>
                  <th>Option</th>
                  <th>Type</th>
                  <th>Default</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>endpoint</code></td>
                  <td>string</td>
                  <td>required</td>
                  <td>URL of your analysis API endpoint</td>
                </tr>
                <tr>
                  <td><code>apiKey</code></td>
                  <td>string</td>
                  <td>null</td>
                  <td>API key for authentication</td>
                </tr>
                <tr>
                  <td><code>sampleRate</code></td>
                  <td>number</td>
                  <td>100</td>
                  <td>Maximum events to capture per second</td>
                </tr>
                <tr>
                  <td><code>batchSize</code></td>
                  <td>number</td>
                  <td>50</td>
                  <td>Events to collect before sending to API</td>
                </tr>
                <tr>
                  <td><code>batchInterval</code></td>
                  <td>number</td>
                  <td>500</td>
                  <td>Milliseconds between API calls</td>
                </tr>
                <tr>
                  <td><code>trackMouse</code></td>
                  <td>boolean</td>
                  <td>true</td>
                  <td>Track mouse movements</td>
                </tr>
                <tr>
                  <td><code>trackKeyboard</code></td>
                  <td>boolean</td>
                  <td>true</td>
                  <td>Track keystroke timing</td>
                </tr>
                <tr>
                  <td><code>trackClicks</code></td>
                  <td>boolean</td>
                  <td>true</td>
                  <td>Track click events</td>
                </tr>
                <tr>
                  <td><code>trackScroll</code></td>
                  <td>boolean</td>
                  <td>true</td>
                  <td>Track scroll behavior</td>
                </tr>
                <tr>
                  <td><code>debug</code></td>
                  <td>boolean</td>
                  <td>false</td>
                  <td>Enable console logging</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Methods */}
      <section className="sdk-methods-section">
        <div className="sdk-container">
          <h2>SDK Methods</h2>
          <div className="sdk-methods-grid">
            <div className="sdk-method-card">
              <h4><code>detector.start()</code></h4>
              <p>Begin capturing user behavior events. Call this when you want to start monitoring.</p>
            </div>
            <div className="sdk-method-card">
              <h4><code>detector.stop()</code></h4>
              <p>Stop capturing events. Useful for cleanup when component unmounts.</p>
            </div>
            <div className="sdk-method-card">
              <h4><code>detector.getVerdict()</code></h4>
              <p>Returns the current verdict synchronously. May be null if not enough data collected.</p>
            </div>
            <div className="sdk-method-card">
              <h4><code>detector.reset()</code></h4>
              <p>Clear all collected data and start fresh. Useful for new sessions.</p>
            </div>
            <div className="sdk-method-card">
              <h4><code>detector.onVerdict(callback)</code></h4>
              <p>Register a callback to be called whenever a new verdict is received.</p>
            </div>
            <div className="sdk-method-card">
              <h4><code>detector.getStats()</code></h4>
              <p>Returns statistics about collected events (count, types, etc.).</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="sdk-cta">
        <div className="sdk-container">
          <div className="sdk-cta-content">
            <h2>Ready to Block Bots?</h2>
            <p>Try the live demo to see how it works, or check the API reference for backend integration.</p>
            <div className="sdk-cta-actions">
              <Link to="/demo" className="btn btn-primary btn-lg">
                <span>Try Live Demo</span>
                <ArrowRight size={18} />
              </Link>
              <Link to="/api" className="btn btn-secondary btn-lg">
                API Reference
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
