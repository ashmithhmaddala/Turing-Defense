import { Link } from 'react-router-dom';
import { 
  Mouse, Keyboard, Target, Activity, Brain, Shield, 
  Zap, Eye, Clock, BarChart, ArrowRight 
} from '../components/Icons';

export default function FeaturesPage() {
  const featureCategories = [
    {
      title: 'Mouse Dynamics',
      icon: Mouse,
      description: 'Advanced analysis of mouse movement patterns',
      features: [
        {
          name: 'Velocity Analysis',
          desc: 'Measures instantaneous and average mouse speed. Bots often move at constant velocities while humans show natural acceleration/deceleration.'
        },
        {
          name: 'Acceleration Patterns',
          desc: 'Tracks changes in velocity over time. Human movements have characteristic acceleration curves when starting and stopping.'
        },
        {
          name: 'Jerk Detection',
          desc: 'Rate of change of acceleration. Human movements have smooth jerk profiles; bots produce jagged, unnatural patterns.'
        },
        {
          name: 'Path Curvature',
          desc: 'Analyzes the geometry of mouse paths. Humans create curved, slightly imprecise paths while bots often move in straight lines.'
        },
        {
          name: 'Direction Changes',
          desc: 'Frequency and angle of direction changes. Natural hesitation and course correction vs. mechanical precision.'
        },
        {
          name: 'Pause Patterns',
          desc: 'Duration and placement of pauses. Humans pause to think; bots typically don\'t pause or pause at regular intervals.'
        }
      ]
    },
    {
      title: 'Keystroke Dynamics',
      icon: Keyboard,
      description: 'Biometric analysis of typing patterns',
      features: [
        {
          name: 'Key Hold Duration',
          desc: 'Time each key is pressed. Humans have variable hold times based on finger strength and key position.'
        },
        {
          name: 'Inter-Key Intervals',
          desc: 'Time between consecutive keystrokes. Human typing has characteristic rhythm variations.'
        },
        {
          name: 'Digraph Timing',
          desc: 'Timing patterns for specific two-key combinations. "th" is faster than "qz" for touch typists.'
        },
        {
          name: 'Typing Rhythm',
          desc: 'Overall cadence and consistency. Humans speed up/slow down; bots are often too consistent.'
        },
        {
          name: 'Error Patterns',
          desc: 'Backspace usage and correction behaviors indicate human uncertainty and mistakes.'
        },
        {
          name: 'Word Boundaries',
          desc: 'Natural pauses at spaces and punctuation reflect human reading and composition.'
        }
      ]
    },
    {
      title: 'Click Analysis',
      icon: Target,
      description: 'Precision and timing of click events',
      features: [
        {
          name: 'Click Precision',
          desc: 'Distribution of clicks relative to targets. Humans show Gaussian distribution; bots hit exact centers.'
        },
        {
          name: 'Double-Click Timing',
          desc: 'Interval between double-clicks. Each human has a characteristic double-click speed.'
        },
        {
          name: 'Click Duration',
          desc: 'Mouse button hold time. Varies naturally in humans based on intent and fatigue.'
        },
        {
          name: 'Pre-Click Movement',
          desc: 'Mouse behavior before clicking. Humans often overshoot and correct; bots move directly.'
        },
        {
          name: 'Post-Click Behavior',
          desc: 'Movement immediately after clicking. Humans have natural micro-movements during clicks.'
        },
        {
          name: 'Click Location Entropy',
          desc: 'Randomness in click positioning. Too perfect or too random both indicate bot behavior.'
        }
      ]
    },
    {
      title: 'Scroll Behavior',
      icon: Activity,
      description: 'Analysis of page scrolling patterns',
      features: [
        {
          name: 'Scroll Velocity',
          desc: 'Speed of scrolling varies naturally in humans based on content engagement.'
        },
        {
          name: 'Scroll Direction Changes',
          desc: 'Humans scroll back up to re-read; bots typically scroll monotonically.'
        },
        {
          name: 'Scroll Momentum',
          desc: 'Natural deceleration patterns from trackpads and scroll wheels.'
        },
        {
          name: 'Reading Pauses',
          desc: 'Stops at content indicate actual reading vs. automated page scanning.'
        },
        {
          name: 'Scroll Depth',
          desc: 'How far users scroll correlates with engagement patterns.'
        },
        {
          name: 'Scroll Frequency',
          desc: 'Timing between scroll actions reveals natural reading rhythm.'
        }
      ]
    }
  ];

  const technicalFeatures = [
    {
      icon: Brain,
      title: 'Neural Network',
      value: '3-Layer',
      desc: 'Deep learning architecture with 128→64→32 neurons for pattern recognition'
    },
    {
      icon: BarChart,
      title: 'Accuracy',
      value: '91.8%',
      desc: 'Validated on research datasets including Balabit and custom adversarial samples'
    },
    {
      icon: Zap,
      title: 'Response Time',
      value: '<50ms',
      desc: 'Real-time inference with minimal latency for seamless user experience'
    },
    {
      icon: Activity,
      title: 'Features',
      value: '56',
      desc: 'Comprehensive behavioral feature extraction across all input modalities'
    },
    {
      icon: Shield,
      title: 'Adversarial Hardening',
      value: 'Yes',
      desc: 'Trained against adversarial examples to resist sophisticated bot attacks'
    },
    {
      icon: Eye,
      title: 'Invisible Detection',
      value: 'Zero UI',
      desc: 'No CAPTCHAs, no puzzles - completely transparent to legitimate users'
    }
  ];

  const comparisonData = [
    { feature: 'User Friction', turingDefense: 'None', captcha: 'High', honeypot: 'None' },
    { feature: 'Bot Detection Rate', turingDefense: '91.8%', captcha: '~85%', honeypot: '~60%' },
    { feature: 'False Positive Rate', turingDefense: '<5%', captcha: '~10%', honeypot: '<1%' },
    { feature: 'Accessibility', turingDefense: 'Full', captcha: 'Limited', honeypot: 'Full' },
    { feature: 'Sophisticated Bot Detection', turingDefense: 'Yes', captcha: 'Sometimes', honeypot: 'No' },
    { feature: 'Real-time Analysis', turingDefense: 'Yes', captcha: 'No', honeypot: 'No' },
    { feature: 'Machine Learning', turingDefense: 'Yes', captcha: 'Some', honeypot: 'No' },
    { feature: 'Privacy Impact', turingDefense: 'Low', captcha: 'High', honeypot: 'None' }
  ];

  return (
    <div className="features-page">
      {/* Hero */}
      <section className="features-hero">
        <div className="features-hero-content">
          <h1>56 Behavioral Features</h1>
          <p>
            Turing Defense analyzes dozens of behavioral signals in real-time to distinguish 
            humans from bots with high accuracy and zero user friction.
          </p>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="features-stats-bar">
        <div className="features-container">
          <div className="stats-bar-grid">
            {technicalFeatures.map((feat, idx) => (
              <div key={idx} className="stat-bar-item">
                <feat.icon size={24} />
                <div className="stat-bar-info">
                  <span className="stat-bar-value">{feat.value}</span>
                  <span className="stat-bar-label">{feat.title}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Feature Categories */}
      {featureCategories.map((category, catIdx) => (
        <section 
          key={catIdx} 
          className={`feature-category-section ${catIdx % 2 === 1 ? 'alt' : ''}`}
        >
          <div className="features-container">
            <div className="category-header">
              <div className="category-icon">
                <category.icon size={32} />
              </div>
              <div className="category-info">
                <h2>{category.title}</h2>
                <p>{category.description}</p>
              </div>
            </div>
            
            <div className="feature-details-grid">
              {category.features.map((feature, featIdx) => (
                <div key={featIdx} className="feature-detail-card">
                  <h4>{feature.name}</h4>
                  <p>{feature.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      ))}

      {/* How It Works */}
      <section className="how-it-works-section">
        <div className="features-container">
          <h2>How Detection Works</h2>
          <div className="process-flow">
            <div className="process-step">
              <div className="step-number">1</div>
              <h4>Capture</h4>
              <p>JavaScript SDK silently captures mouse, keyboard, click, and scroll events</p>
            </div>
            <div className="process-arrow">→</div>
            <div className="process-step">
              <div className="step-number">2</div>
              <h4>Extract</h4>
              <p>56 behavioral features computed from raw event data in real-time</p>
            </div>
            <div className="process-arrow">→</div>
            <div className="process-step">
              <div className="step-number">3</div>
              <h4>Analyze</h4>
              <p>Neural network processes features and outputs human probability score</p>
            </div>
            <div className="process-arrow">→</div>
            <div className="process-step">
              <div className="step-number">4</div>
              <h4>Decide</h4>
              <p>Verdict returned with confidence level and trigger explanations</p>
            </div>
          </div>
        </div>
      </section>

      {/* Comparison Table */}
      <section className="comparison-section">
        <div className="features-container">
          <h2>How We Compare</h2>
          <p className="section-desc">
            See how behavioral detection stacks up against traditional bot prevention methods.
          </p>
          
          <div className="comparison-table-wrapper">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th className="highlight">Turing Defense</th>
                  <th>CAPTCHA</th>
                  <th>Honeypot</th>
                </tr>
              </thead>
              <tbody>
                {comparisonData.map((row, idx) => (
                  <tr key={idx}>
                    <td>{row.feature}</td>
                    <td className="highlight">{row.turingDefense}</td>
                    <td>{row.captcha}</td>
                    <td>{row.honeypot}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Neural Network Architecture */}
      <section className="architecture-section">
        <div className="features-container">
          <h2>Neural Network Architecture</h2>
          <div className="architecture-visual">
            <div className="arch-layer input">
              <div className="layer-label">Input</div>
              <div className="layer-neurons">56 features</div>
            </div>
            <div className="arch-connection">→</div>
            <div className="arch-layer hidden">
              <div className="layer-label">Hidden 1</div>
              <div className="layer-neurons">128 neurons</div>
              <div className="layer-extra">ReLU + Dropout(0.3)</div>
            </div>
            <div className="arch-connection">→</div>
            <div className="arch-layer hidden">
              <div className="layer-label">Hidden 2</div>
              <div className="layer-neurons">64 neurons</div>
              <div className="layer-extra">ReLU + Dropout(0.3)</div>
            </div>
            <div className="arch-connection">→</div>
            <div className="arch-layer hidden">
              <div className="layer-label">Hidden 3</div>
              <div className="layer-neurons">32 neurons</div>
              <div className="layer-extra">ReLU</div>
            </div>
            <div className="arch-connection">→</div>
            <div className="arch-layer output">
              <div className="layer-label">Output</div>
              <div className="layer-neurons">1 (sigmoid)</div>
              <div className="layer-extra">P(human)</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="features-cta">
        <div className="features-container">
          <div className="features-cta-content">
            <h2>See It In Action</h2>
            <p>Try the interactive demo to see how behavioral analysis detects bots in real-time.</p>
            <div className="features-cta-actions">
              <Link to="/demo" className="btn btn-primary btn-lg">
                <span>Try Live Demo</span>
                <ArrowRight size={18} />
              </Link>
              <Link to="/sdk" className="btn btn-secondary btn-lg">
                View SDK
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
