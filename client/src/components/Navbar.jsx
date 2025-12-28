import { Link, useLocation } from 'react-router-dom';
import { useState } from 'react';
import { TuringLogo, Github, Menu, X } from './Icons';

export default function Navbar() {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  const isActive = (path) => location.pathname === path;
  
  const navLinks = [
    { path: '/', label: 'Home' },
    { path: '/demo', label: 'Demo' },
    { path: '/features', label: 'Features' },
    { path: '/docs', label: 'Docs' },
    { path: '/sdk', label: 'SDK' },
    { path: '/api', label: 'API' },
    { path: '/about', label: 'About' },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-brand">
          <div className="brand-icon">
            <TuringLogo size={28} />
          </div>
          <span className="brand-text">Turing Defense</span>
        </Link>
        
        <div className={`navbar-links ${mobileMenuOpen ? 'open' : ''}`}>
          {navLinks.map(link => (
            <Link 
              key={link.path}
              to={link.path} 
              className={`nav-link ${isActive(link.path) ? 'active' : ''}`}
              onClick={() => setMobileMenuOpen(false)}
            >
              {link.label}
            </Link>
          ))}
          <a 
            href="https://github.com/ashmithhmaddala/Turing-Defense" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="nav-link github-link"
          >
            <Github size={18} />
            <span>GitHub</span>
          </a>
        </div>
        
        <button 
          className="mobile-menu-btn"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
    </nav>
  );
}
