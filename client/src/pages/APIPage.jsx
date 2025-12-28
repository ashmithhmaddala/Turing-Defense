import { useState } from 'react';
import { Copy, Check, Terminal, Code, ArrowRight } from '../components/Icons';
import { Link } from 'react-router-dom';

export default function APIPage() {
  const [copiedIndex, setCopiedIndex] = useState(null);

  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const endpoints = [
    {
      method: 'POST',
      path: '/analyze',
      description: 'Analyze behavioral data and return a bot detection verdict',
      requestBody: `{
  "events": [
    {
      "type": "mousemove",
      "x": 234,
      "y": 567,
      "timestamp": 1703789012345
    },
    {
      "type": "keydown",
      "key": "a",
      "timestamp": 1703789012456
    },
    {
      "type": "click",
      "x": 400,
      "y": 300,
      "timestamp": 1703789012567
    }
  ]
}`,
      responseBody: `{
  "score": 0.87,
  "verdict": "human",
  "confidence": 0.92,
  "triggers": [
    {
      "type": "success",
      "message": "Natural mouse movement patterns"
    },
    {
      "type": "success", 
      "message": "Human-like typing rhythm"
    }
  ],
  "features": {
    "mouse_velocity_mean": 0.34,
    "mouse_acceleration_std": 0.12,
    "keystroke_interval_mean": 0.15,
    "click_precision": 0.89
  }
}`,
      curlExample: `curl -X POST http://localhost:5000/analyze \\
  -H "Content-Type: application/json" \\
  -d '{
    "events": [
      {"type": "mousemove", "x": 100, "y": 200, "timestamp": 1703789012345},
      {"type": "mousemove", "x": 105, "y": 203, "timestamp": 1703789012355},
      {"type": "click", "x": 105, "y": 203, "timestamp": 1703789012400}
    ]
  }'`
    },
    {
      method: 'GET',
      path: '/health',
      description: 'Check if the API server is running and healthy',
      requestBody: null,
      responseBody: `{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime": 3600
}`,
      curlExample: `curl http://localhost:5000/health`
    },
    {
      method: 'GET',
      path: '/model/info',
      description: 'Get information about the loaded neural network model',
      requestBody: null,
      responseBody: `{
  "model_type": "NeuralBotDetector",
  "accuracy": 0.918,
  "features": 56,
  "architecture": {
    "layers": [
      {"type": "Linear", "in": 56, "out": 128},
      {"type": "ReLU"},
      {"type": "Dropout", "p": 0.3},
      {"type": "Linear", "in": 128, "out": 64},
      {"type": "ReLU"},
      {"type": "Dropout", "p": 0.3},
      {"type": "Linear", "in": 64, "out": 32},
      {"type": "ReLU"},
      {"type": "Linear", "in": 32, "out": 1},
      {"type": "Sigmoid"}
    ]
  },
  "training": {
    "epochs": 100,
    "batch_size": 64,
    "optimizer": "Adam"
  }
}`,
      curlExample: `curl http://localhost:5000/model/info`
    }
  ];

  const eventTypes = [
    {
      type: 'mousemove',
      fields: [
        { name: 'type', type: 'string', desc: '"mousemove"' },
        { name: 'x', type: 'number', desc: 'X coordinate in pixels' },
        { name: 'y', type: 'number', desc: 'Y coordinate in pixels' },
        { name: 'timestamp', type: 'number', desc: 'Unix timestamp in milliseconds' }
      ]
    },
    {
      type: 'click',
      fields: [
        { name: 'type', type: 'string', desc: '"click"' },
        { name: 'x', type: 'number', desc: 'X coordinate of click' },
        { name: 'y', type: 'number', desc: 'Y coordinate of click' },
        { name: 'button', type: 'number', desc: '0=left, 1=middle, 2=right' },
        { name: 'timestamp', type: 'number', desc: 'Unix timestamp in milliseconds' }
      ]
    },
    {
      type: 'keydown',
      fields: [
        { name: 'type', type: 'string', desc: '"keydown"' },
        { name: 'key', type: 'string', desc: 'Key pressed (e.g., "a", "Enter")' },
        { name: 'timestamp', type: 'number', desc: 'Unix timestamp in milliseconds' }
      ]
    },
    {
      type: 'keyup',
      fields: [
        { name: 'type', type: 'string', desc: '"keyup"' },
        { name: 'key', type: 'string', desc: 'Key released' },
        { name: 'timestamp', type: 'number', desc: 'Unix timestamp in milliseconds' }
      ]
    },
    {
      type: 'scroll',
      fields: [
        { name: 'type', type: 'string', desc: '"scroll"' },
        { name: 'scrollX', type: 'number', desc: 'Horizontal scroll position' },
        { name: 'scrollY', type: 'number', desc: 'Vertical scroll position' },
        { name: 'timestamp', type: 'number', desc: 'Unix timestamp in milliseconds' }
      ]
    }
  ];

  const responseCodes = [
    { code: 200, status: 'OK', description: 'Request successful, verdict returned' },
    { code: 400, status: 'Bad Request', description: 'Invalid request body or missing required fields' },
    { code: 422, status: 'Unprocessable Entity', description: 'Not enough events to make a determination' },
    { code: 500, status: 'Internal Server Error', description: 'Server error during processing' }
  ];

  return (
    <div className="api-page">
      {/* Hero */}
      <section className="api-hero">
        <div className="api-hero-content">
          <div className="api-badge">
            <Terminal size={16} />
            <span>REST API</span>
          </div>
          <h1>API Reference</h1>
          <p>
            Complete reference for the Turing Defense REST API. 
            Send behavioral events, receive bot detection verdicts.
          </p>
        </div>
      </section>

      {/* Base URL */}
      <section className="api-section">
        <div className="api-container">
          <h2>Base URL</h2>
          <div className="api-base-url">
            <code>http://localhost:5000</code>
            <p>For production, replace with your deployed API URL.</p>
          </div>
        </div>
      </section>

      {/* Endpoints */}
      <section className="api-section">
        <div className="api-container">
          <h2>Endpoints</h2>
          
          {endpoints.map((endpoint, idx) => (
            <div key={idx} className="api-endpoint" id={endpoint.path.replace('/', '')}>
              <div className="endpoint-header">
                <span className={`method ${endpoint.method.toLowerCase()}`}>
                  {endpoint.method}
                </span>
                <code className="endpoint-path">{endpoint.path}</code>
              </div>
              <p className="endpoint-desc">{endpoint.description}</p>
              
              {endpoint.requestBody && (
                <div className="endpoint-section">
                  <h4>Request Body</h4>
                  <div className="code-block">
                    <div className="code-header">
                      <span>JSON</span>
                      <button 
                        className="copy-btn"
                        onClick={() => copyToClipboard(endpoint.requestBody, `req-${idx}`)}
                      >
                        {copiedIndex === `req-${idx}` ? <Check size={14} /> : <Copy size={14} />}
                        <span>{copiedIndex === `req-${idx}` ? 'Copied!' : 'Copy'}</span>
                      </button>
                    </div>
                    <pre><code>{endpoint.requestBody}</code></pre>
                  </div>
                </div>
              )}
              
              <div className="endpoint-section">
                <h4>Response</h4>
                <div className="code-block">
                  <div className="code-header">
                    <span>JSON</span>
                    <button 
                      className="copy-btn"
                      onClick={() => copyToClipboard(endpoint.responseBody, `res-${idx}`)}
                    >
                      {copiedIndex === `res-${idx}` ? <Check size={14} /> : <Copy size={14} />}
                      <span>{copiedIndex === `res-${idx}` ? 'Copied!' : 'Copy'}</span>
                    </button>
                  </div>
                  <pre><code>{endpoint.responseBody}</code></pre>
                </div>
              </div>

              <div className="endpoint-section">
                <h4>Example</h4>
                <div className="code-block">
                  <div className="code-header">
                    <span>cURL</span>
                    <button 
                      className="copy-btn"
                      onClick={() => copyToClipboard(endpoint.curlExample, `curl-${idx}`)}
                    >
                      {copiedIndex === `curl-${idx}` ? <Check size={14} /> : <Copy size={14} />}
                      <span>{copiedIndex === `curl-${idx}` ? 'Copied!' : 'Copy'}</span>
                    </button>
                  </div>
                  <pre><code>{endpoint.curlExample}</code></pre>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Event Types */}
      <section className="api-section alt">
        <div className="api-container">
          <h2>Event Types</h2>
          <p className="section-desc">
            The API accepts the following event types. Each event must include a timestamp.
          </p>
          
          <div className="event-types-grid">
            {eventTypes.map((event, idx) => (
              <div key={idx} className="event-type-card">
                <h4><code>{event.type}</code></h4>
                <table className="event-fields-table">
                  <thead>
                    <tr>
                      <th>Field</th>
                      <th>Type</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {event.fields.map((field, fidx) => (
                      <tr key={fidx}>
                        <td><code>{field.name}</code></td>
                        <td>{field.type}</td>
                        <td>{field.desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Response Codes */}
      <section className="api-section">
        <div className="api-container">
          <h2>Response Codes</h2>
          <div className="response-codes-table-wrapper">
            <table className="response-codes-table">
              <thead>
                <tr>
                  <th>Code</th>
                  <th>Status</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {responseCodes.map((rc, idx) => (
                  <tr key={idx}>
                    <td><code>{rc.code}</code></td>
                    <td>{rc.status}</td>
                    <td>{rc.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Verdict Values */}
      <section className="api-section alt">
        <div className="api-container">
          <h2>Verdict Values</h2>
          <div className="verdict-grid">
            <div className="verdict-card human">
              <h4>human</h4>
              <p>Score ≥ 0.6. The behavior patterns indicate a human user.</p>
            </div>
            <div className="verdict-card uncertain">
              <h4>uncertain</h4>
              <p>Score between 0.4 and 0.6. Not enough confidence to determine.</p>
            </div>
            <div className="verdict-card bot">
              <h4>bot</h4>
              <p>Score &lt; 0.4. The behavior patterns indicate automated activity.</p>
            </div>
          </div>
        </div>
      </section>

      {/* WebSocket */}
      <section className="api-section">
        <div className="api-container">
          <h2>WebSocket API</h2>
          <p className="section-desc">
            For real-time detection, connect via WebSocket for streaming analysis.
          </p>
          
          <div className="websocket-info">
            <div className="ws-endpoint">
              <h4>Connection URL</h4>
              <code>ws://localhost:5000/socket.io</code>
            </div>
            
            <div className="ws-events">
              <h4>Events</h4>
              <div className="ws-event">
                <code className="ws-emit">emit</code>
                <code>behavior_data</code>
                <span>Send behavioral events for analysis</span>
              </div>
              <div className="ws-event">
                <code className="ws-on">on</code>
                <code>analysis_result</code>
                <span>Receive real-time verdict updates</span>
              </div>
              <div className="ws-event">
                <code className="ws-on">on</code>
                <code>connect</code>
                <span>Connection established</span>
              </div>
            </div>
          </div>

          <div className="code-block">
            <div className="code-header">
              <span>JavaScript WebSocket Example</span>
            </div>
            <pre><code>{`import { io } from 'socket.io-client';

const socket = io('http://localhost:5000');

socket.on('connect', () => {
  console.log('Connected to Turing Defense');
});

// Send events
socket.emit('behavior_data', {
  events: collectedEvents,
  timestamp: Date.now()
});

// Receive verdicts
socket.on('analysis_result', (result) => {
  console.log('Score:', result.score);
  console.log('Verdict:', result.verdict);
});`}</code></pre>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="api-cta">
        <div className="api-container">
          <div className="api-cta-content">
            <h2>Ready to Integrate?</h2>
            <p>Use our JavaScript SDK for easy client-side integration, or try the live demo.</p>
            <div className="api-cta-actions">
              <Link to="/sdk" className="btn btn-primary btn-lg">
                <Code size={18} />
                <span>View SDK</span>
              </Link>
              <Link to="/demo" className="btn btn-secondary btn-lg">
                Try Demo
                <ArrowRight size={18} />
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
