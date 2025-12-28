import { Link } from 'react-router-dom';
import { TuringLogo, Github } from './Icons';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-main">
          <div className="footer-brand-section">
            <Link to="/" className="footer-brand">
              <TuringLogo size={24} />
              <span>Turing Defense</span>
            </Link>
            <p className="footer-tagline">
              Invisible bot detection through behavioral analysis
            </p>
          </div>
          
          <div className="footer-links-grid">
            <div className="footer-column">
              <h4>Product</h4>
              <Link to="/demo">Live Demo</Link>
              <Link to="/features">Features</Link>
              <Link to="/sdk">SDK</Link>
            </div>
            <div className="footer-column">
              <h4>Resources</h4>
              <Link to="/docs">Documentation</Link>
              <Link to="/api">API Reference</Link>
              <Link to="/about">About</Link>
            </div>
            <div className="footer-column">
              <h4>Connect</h4>
              <a href="https://github.com/ashmithhmaddala/Turing-Defense" target="_blank" rel="noopener noreferrer">
                GitHub
              </a>
              <a href="https://www.linkedin.com/in/ashmith-maddala/" target="_blank" rel="noopener noreferrer">
                LinkedIn
              </a>
            </div>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p>Built by <strong>Ashmith Maddala</strong></p>
          <a 
            href="https://github.com/ashmithhmaddala/Turing-Defense" 
            target="_blank" 
            rel="noopener noreferrer"
            className="footer-github"
          >
            <Github size={16} />
            <span>View Source</span>
          </a>
        </div>
      </div>
    </footer>
  );
}
