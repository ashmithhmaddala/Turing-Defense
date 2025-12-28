import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Shield, Zap, Eye, Code, ArrowRight, Mouse, 
  Keyboard, Target, Brain, BarChart, Clock 
} from '../components/Icons';

export default function HomePage() {
  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-bg">
          <div className="hero-gradient" />
          <div className="hero-grid" />
        </div>
        
        <div className="hero-content">
          <motion.div 
            className="hero-badge"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <span className="badge-dot" />
            <span>Open Source Bot Detection</span>
          </motion.div>
          
          <motion.h1 
            className="hero-title"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            Detect Bots Through
            <span className="gradient-text"> Behavioral Analysis</span>
          </motion.h1>
          
          <motion.p 
            className="hero-description"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            Turing Defense uses machine learning to analyze mouse movements, keystrokes, 
            and interaction patterns in real-time. No CAPTCHAs, no friction, just invisible protection.
          </motion.p>
          
          <motion.div 
            className="hero-actions"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Link to="/demo" className="btn btn-primary btn-lg">
              <span>Try Live Demo</span>
              <ArrowRight size={20} />
            </Link>
            <Link to="/docs" className="btn btn-secondary btn-lg">
              <Code size={20} />
              <span>View Documentation</span>
            </Link>
          </motion.div>
          
          <motion.div 
            className="hero-stats"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <div className="stat">
              <span className="stat-value">91.8%</span>
              <span className="stat-label">Accuracy</span>
            </div>
            <div className="stat-divider" />
            <div className="stat">
              <span className="stat-value">56</span>
              <span className="stat-label">Features Analyzed</span>
            </div>
            <div className="stat-divider" />
            <div className="stat">
              <span className="stat-value">&lt;50ms</span>
              <span className="stat-label">Response Time</span>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="section-container">
          <div className="section-header">
            <h2>How It Works</h2>
            <p>Analyze behavioral patterns that distinguish humans from automated scripts</p>
          </div>
          
          <div className="features-grid">
            <motion.div 
              className="feature-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <div className="feature-icon">
                <Mouse size={32} />
              </div>
              <h3>Mouse Dynamics</h3>
              <p>
                Track velocity, acceleration, path curvature, and micro-corrections. 
                Humans have natural hesitation and overshoot patterns that bots cannot replicate.
              </p>
            </motion.div>
            
            <motion.div 
              className="feature-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <div className="feature-icon">
                <Keyboard size={32} />
              </div>
              <h3>Keystroke Analysis</h3>
              <p>
                Measure typing rhythm, key hold duration, and inter-key timing. 
                Every person has a unique typing signature that is nearly impossible to fake.
              </p>
            </motion.div>
            
            <motion.div 
              className="feature-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="feature-icon">
                <Target size={32} />
              </div>
              <h3>Click Patterns</h3>
              <p>
                Analyze click precision, timing, and sequences. Bots often click with 
                unnatural precision or follow predictable patterns.
              </p>
            </motion.div>
            
            <motion.div 
              className="feature-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <div className="feature-icon">
                <Brain size={32} />
              </div>
              <h3>Neural Network</h3>
              <p>
                A trained PyTorch model processes 56 behavioral features to generate 
                a confidence score distinguishing humans from bots.
              </p>
            </motion.div>
            
            <motion.div 
              className="feature-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <div className="feature-icon">
                <Clock size={32} />
              </div>
              <h3>Real-Time Analysis</h3>
              <p>
                WebSocket connection enables continuous monitoring with instant feedback. 
                Detection happens in under 50 milliseconds.
              </p>
            </motion.div>
            
            <motion.div 
              className="feature-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.5 }}
            >
              <div className="feature-icon">
                <BarChart size={32} />
              </div>
              <h3>Detection Signals</h3>
              <p>
                Get detailed explanations of why a user was flagged, including 
                specific anomalies in velocity, straightness, or timing.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Why Section */}
      <section className="why-section">
        <div className="section-container">
          <div className="why-grid">
            <div className="why-content">
              <h2>Why Behavioral Detection?</h2>
              <p className="why-lead">
                Traditional CAPTCHAs frustrate users and can be bypassed by modern bots. 
                Behavioral analysis works invisibly in the background.
              </p>
              
              <div className="why-points">
                <div className="why-point">
                  <Shield size={24} />
                  <div>
                    <h4>Invisible Protection</h4>
                    <p>No puzzles, no friction. Users never know they are being verified.</p>
                  </div>
                </div>
                <div className="why-point">
                  <Zap size={24} />
                  <div>
                    <h4>Continuous Monitoring</h4>
                    <p>Unlike one-time CAPTCHAs, behavioral analysis runs throughout the session.</p>
                  </div>
                </div>
                <div className="why-point">
                  <Eye size={24} />
                  <div>
                    <h4>Hard to Spoof</h4>
                    <p>Replicating human-like behavior is computationally expensive for attackers.</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="why-visual">
              <div className="comparison-card">
                <div className="comparison-item bad">
                  <h4>Traditional CAPTCHAs</h4>
                  <ul>
                    <li>Frustrate real users</li>
                    <li>One-time verification only</li>
                    <li>Solved by ML services</li>
                    <li>Accessibility issues</li>
                  </ul>
                </div>
                <div className="comparison-item good">
                  <h4>Behavioral Analysis</h4>
                  <ul>
                    <li>Invisible to users</li>
                    <li>Continuous protection</li>
                    <li>Hard to automate</li>
                    <li>Works for everyone</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="section-container">
          <div className="cta-content">
            <h2>See It In Action</h2>
            <p>
              Try the live demo to see how your own behavior is analyzed in real-time. 
              Toggle the bot simulator to watch detection scores change instantly.
            </p>
            <Link to="/demo" className="btn btn-primary btn-lg">
              <span>Launch Demo</span>
              <ArrowRight size={20} />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
