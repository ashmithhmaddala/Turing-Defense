import { useState } from 'react';
import { Code, Terminal, Book, Shield, Copy, Check } from '../components/Icons';

const CodeBlock = ({ code, language = 'javascript' }) => {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  return (
    <div className="code-block">
      <div className="code-header">
        <span>{language}</span>
        <button onClick={handleCopy} className="copy-btn">
          {copied ? <><Check size={14} /> Copied</> : <><Copy size={14} /> Copy</>}
        </button>
      </div>
      <pre><code>{code}</code></pre>
    </div>
  );
};

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState('overview');
  
  const sections = [
    { id: 'overview', label: 'Overview' },
    { id: 'quickstart', label: 'Quick Start' },
    { id: 'sdk', label: 'JavaScript SDK' },
    { id: 'api', label: 'API Reference' },
    { id: 'features', label: 'Features' },
    { id: 'architecture', label: 'Architecture' },
  ];

  return (
    <div className="docs-page">
      {/* Sidebar */}
      <aside className="docs-sidebar">
        <nav className="docs-nav">
          <h4>Documentation</h4>
          {sections.map(section => (
            <a
              key={section.id}
              href={`#${section.id}`}
              className={`docs-nav-item ${activeSection === section.id ? 'active' : ''}`}
              onClick={() => setActiveSection(section.id)}
            >
              {section.label}
            </a>
          ))}
        </nav>
      </aside>
      
      {/* Main Content */}
      <main className="docs-content">
        <section id="overview" className="docs-section">
          <h1>Turing Defense Documentation</h1>
          <p className="docs-lead">
            Turing Defense is an open-source bot detection system that uses behavioral biometrics 
            to distinguish humans from automated scripts in real-time.
          </p>
          
          <div className="docs-cards">
            <div className="docs-card">
              <Shield size={24} />
              <h3>Invisible Protection</h3>
              <p>No CAPTCHAs or user friction. Detection happens silently in the background.</p>
            </div>
            <div className="docs-card">
              <Terminal size={24} />
              <h3>Easy Integration</h3>
              <p>Drop-in JavaScript SDK works with any website. Backend API for custom implementations.</p>
            </div>
            <div className="docs-card">
              <Book size={24} />
              <h3>Well Documented</h3>
              <p>Comprehensive guides, API reference, and examples to get you started quickly.</p>
            </div>
          </div>
        </section>

        <section id="quickstart" className="docs-section">
          <h2>Quick Start</h2>
          <p>Get Turing Defense running locally in under 5 minutes.</p>
          
          <h3>Prerequisites</h3>
          <ul>
            <li>Python 3.8+</li>
            <li>Node.js 18+</li>
            <li>Git</li>
          </ul>
          
          <h3>Installation</h3>
          <CodeBlock 
            language="bash"
            code={`# Clone the repository
git clone https://github.com/ashmithhmaddala/Turing-Defense.git
cd Turing-Defense

# Start the backend
cd backend
pip install -r requirements.txt
python app.py

# In a new terminal, start the frontend
cd client
npm install
npm run dev`}
          />
          
          <p>
            The demo will be available at <code>http://localhost:5174</code> with the 
            backend API running on <code>http://localhost:5000</code>.
          </p>
        </section>

        <section id="sdk" className="docs-section">
          <h2>JavaScript SDK</h2>
          <p>
            The SDK provides a simple way to add bot detection to any website. It automatically 
            tracks user behavior and communicates with the detection backend.
          </p>
          
          <h3>Installation</h3>
          <CodeBlock 
            language="html"
            code={`<!-- Add to your HTML -->
<script src="https://your-domain.com/sdk/turing-defense.js"></script>`}
          />
          
          <h3>Basic Usage</h3>
          <CodeBlock 
            language="javascript"
            code={`// Initialize the detector
const detector = new TuringDefense({
  endpoint: 'https://your-backend.com',
  siteId: 'your-site-id',
  debug: false
});

// Start tracking
detector.start();

// Get detection result
detector.onResult((result) => {
  console.log('Bot Score:', result.bot_score);
  console.log('Is Bot:', result.is_bot);
  console.log('Confidence:', result.confidence);
});

// Stop tracking when done
detector.stop();`}
          />
          
          <h3>Configuration Options</h3>
          <div className="docs-table-wrapper">
            <table className="docs-table">
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
                  <td>Backend API URL</td>
                </tr>
                <tr>
                  <td><code>siteId</code></td>
                  <td>string</td>
                  <td>required</td>
                  <td>Your site identifier</td>
                </tr>
                <tr>
                  <td><code>debug</code></td>
                  <td>boolean</td>
                  <td>false</td>
                  <td>Enable console logging</td>
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
                  <td>Track keystrokes</td>
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
                  <td>Track scroll events</td>
                </tr>
                <tr>
                  <td><code>sendInterval</code></td>
                  <td>number</td>
                  <td>500</td>
                  <td>Data send interval (ms)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section id="api" className="docs-section">
          <h2>API Reference</h2>
          <p>The backend exposes both WebSocket and REST endpoints for bot detection.</p>
          
          <h3>WebSocket API</h3>
          <p>Connect to <code>ws://your-backend.com</code> using Socket.IO.</p>
          
          <h4>Events</h4>
          <CodeBlock 
            language="javascript"
            code={`// Send behavior data
socket.emit('behavior_data', {
  mouse_data: [
    { x: 100, y: 200, timestamp: 1703800000000 },
    // ... more points
  ],
  keyboard_data: [
    { key: 'a', timestamp: 1703800000100, type: 'keydown' },
    // ... more events
  ],
  click_data: [
    { x: 150, y: 250, timestamp: 1703800000200 },
  ],
  scroll_data: [
    { deltaY: 100, timestamp: 1703800000300 },
  ]
});

// Receive analysis results
socket.on('analysis_result', (result) => {
  // result.prediction.bot_score (0-100)
  // result.prediction.is_bot (boolean)
  // result.prediction.confidence (0-100)
  // result.prediction.triggers (array of detection signals)
});

// Reset session
socket.emit('reset_session');`}
          />
          
          <h3>REST API</h3>
          
          <div className="api-endpoint">
            <div className="endpoint-header">
              <span className="method post">POST</span>
              <code>/api/analyze</code>
            </div>
            <p>Analyze behavioral data and return bot detection results.</p>
            <h4>Request Body</h4>
            <CodeBlock 
              language="json"
              code={`{
  "site_id": "your-site-id",
  "session_id": "unique-session-id",
  "behavior_data": {
    "mouse_data": [...],
    "keyboard_data": [...],
    "click_data": [...],
    "scroll_data": [...]
  }
}`}
            />
            <h4>Response</h4>
            <CodeBlock 
              language="json"
              code={`{
  "success": true,
  "prediction": {
    "bot_score": 15.3,
    "is_bot": false,
    "confidence": 87.2,
    "triggers": [
      { "type": "success", "text": "Human-like behavior confirmed" },
      { "type": "info", "text": "Natural movement patterns" }
    ]
  }
}`}
            />
          </div>
        </section>

        <section id="features" className="docs-section">
          <h2>Detection Features</h2>
          <p>The neural network analyzes 56 behavioral features across multiple categories.</p>
          
          <div className="features-table">
            <h3>Mouse Dynamics</h3>
            <ul>
              <li><strong>Velocity statistics</strong> - Mean, std, max velocity and coefficient of variation</li>
              <li><strong>Acceleration patterns</strong> - Mean, std, max acceleration</li>
              <li><strong>Jerk analysis</strong> - Rate of change of acceleration</li>
              <li><strong>Path geometry</strong> - Curvature, straightness index, direction changes</li>
              <li><strong>Micro-corrections</strong> - Small adjustments humans naturally make</li>
            </ul>
            
            <h3>Keystroke Biometrics</h3>
            <ul>
              <li><strong>Key hold duration</strong> - How long each key is pressed</li>
              <li><strong>Inter-key intervals</strong> - Time between consecutive keystrokes</li>
              <li><strong>Typing rhythm</strong> - Patterns in typing speed</li>
              <li><strong>Digraph timing</strong> - Timing for common letter pairs</li>
            </ul>
            
            <h3>Click Behavior</h3>
            <ul>
              <li><strong>Click timing</strong> - Distribution of click intervals</li>
              <li><strong>Double-click patterns</strong> - Timing and accuracy</li>
              <li><strong>Click precision</strong> - Accuracy of targeting</li>
            </ul>
            
            <h3>Scroll Patterns</h3>
            <ul>
              <li><strong>Scroll velocity</strong> - Speed and smoothness</li>
              <li><strong>Direction changes</strong> - Reading pattern indicators</li>
              <li><strong>Momentum</strong> - Natural deceleration patterns</li>
            </ul>
          </div>
        </section>

        <section id="architecture" className="docs-section">
          <h2>Architecture</h2>
          <p>Technical overview of the Turing Defense system.</p>
          
          <h3>System Components</h3>
          <div className="architecture-diagram">
            <div className="arch-component">
              <h4>Frontend SDK</h4>
              <p>JavaScript library that captures behavioral events and sends them to the backend via WebSocket.</p>
            </div>
            <div className="arch-arrow">→</div>
            <div className="arch-component">
              <h4>Flask Backend</h4>
              <p>Python server with Flask-SocketIO that processes incoming data and manages sessions.</p>
            </div>
            <div className="arch-arrow">→</div>
            <div className="arch-component">
              <h4>Neural Network</h4>
              <p>PyTorch model (3 layers: 128→64→32) trained on research datasets with adversarial hardening.</p>
            </div>
          </div>
          
          <h3>Model Training</h3>
          <p>The neural network was trained on:</p>
          <ul>
            <li><strong>CMU Keystroke Dataset</strong> - Timing patterns from human typing</li>
            <li><strong>Balabit Mouse Dynamics</strong> - Mouse movement trajectories</li>
            <li><strong>Synthetic bot data</strong> - Generated from common automation frameworks</li>
          </ul>
          
          <h3>Adversarial Robustness</h3>
          <p>
            The model was hardened against evasion attacks using FGSM (Fast Gradient Sign Method) 
            and PGD (Projected Gradient Descent) adversarial training. This makes it resistant to 
            bots that try to mimic human behavior patterns.
          </p>
        </section>
      </main>
    </div>
  );
}
