import { Github, Shield, Brain, Code } from '../components/Icons';

export default function AboutPage() {
  return (
    <div className="about-page">
      <section className="about-hero">
        <div className="about-hero-content">
          <h1>About Turing Defense</h1>
          <p className="about-lead">
            An exploration into behavioral biometrics and invisible bot detection, 
            built to understand how websites can protect themselves without frustrating real users.
          </p>
        </div>
      </section>

      <section className="about-section">
        <div className="about-container">
          <div className="about-story">
            <h2>The Story</h2>
            <p>
              I built Turing Defense because I was curious about how websites detect bots without 
              using CAPTCHAs. We've all been annoyed by "select all the traffic lights" puzzles, 
              but modern security systems often work invisibly in the background. I wanted to 
              understand how.
            </p>
            <p>
              The core insight is simple: humans and bots move differently. We hesitate, overshoot, 
              make tiny corrections. We type with rhythm and personality. Bots, even sophisticated ones, 
              tend to be either too perfect or too random. A machine learning model can pick up on 
              these subtle differences.
            </p>
            <p>
              What started as a weekend project to understand behavioral biometrics turned into a 
              complete detection system. I trained a neural network on research datasets, added 
              adversarial hardening to resist evasion attempts, and built a real-time demo to 
              visualize the detection in action.
            </p>
          </div>

          <div className="about-author">
            <div className="author-card">
              <div className="author-avatar">AM</div>
              <div className="author-info">
                <h3>Ashmith Maddala</h3>
                <p>Creator & Developer</p>
                <div className="author-links">
                  <a 
                    href="https://github.com/ashmithhmaddala" 
                    target="_blank" 
                    rel="noopener noreferrer"
                  >
                    <Github size={18} />
                    <span>GitHub</span>
                  </a>
                  <a 
                    href="https://www.linkedin.com/in/ashmith-maddala/" 
                    target="_blank" 
                    rel="noopener noreferrer"
                  >
                    <span>LinkedIn</span>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="about-section alt">
        <div className="about-container">
          <h2>Technical Highlights</h2>
          <div className="highlights-grid">
            <div className="highlight-card">
              <Brain size={32} />
              <h3>Neural Network</h3>
              <p>
                A 3-layer PyTorch model (128→64→32 neurons) with BatchNorm and Dropout. 
                Trained on CMU Keystroke and Balabit Mouse Dynamics research datasets.
              </p>
            </div>
            <div className="highlight-card">
              <Shield size={32} />
              <h3>Adversarial Hardening</h3>
              <p>
                Trained with FGSM and PGD attacks to resist evasion. Tested against 11 bot 
                profiles including Selenium, Puppeteer, and AI-powered mimicry.
              </p>
            </div>
            <div className="highlight-card">
              <Code size={32} />
              <h3>56 Features</h3>
              <p>
                Analyzes velocity, acceleration, jerk, path geometry, curvature entropy, 
                keystroke timing, click precision, and scroll patterns.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="about-section">
        <div className="about-container">
          <h2>Tech Stack</h2>
          <div className="tech-grid">
            <div className="tech-category">
              <h4>Backend</h4>
              <ul>
                <li>Python 3.11</li>
                <li>Flask + Flask-SocketIO</li>
                <li>PyTorch</li>
                <li>NumPy / SciPy</li>
              </ul>
            </div>
            <div className="tech-category">
              <h4>Frontend</h4>
              <ul>
                <li>React 19</li>
                <li>Vite</li>
                <li>Socket.IO Client</li>
                <li>Framer Motion</li>
              </ul>
            </div>
            <div className="tech-category">
              <h4>Infrastructure</h4>
              <ul>
                <li>Docker</li>
                <li>WebSocket</li>
                <li>REST API</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="about-section alt">
        <div className="about-container center">
          <h2>Open Source</h2>
          <p className="about-lead">
            Turing Defense is open source and available on GitHub. 
            Feel free to explore the code, contribute, or use it in your own projects.
          </p>
          <a 
            href="https://github.com/ashmithhmaddala/Turing-Defense" 
            target="_blank" 
            rel="noopener noreferrer"
            className="btn btn-primary btn-lg"
          >
            <Github size={20} />
            <span>View on GitHub</span>
          </a>
        </div>
      </section>
    </div>
  );
}
