import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import DemoPage from './pages/DemoPage';
import DocsPage from './pages/DocsPage';
import AboutPage from './pages/AboutPage';
import SDKPage from './pages/SDKPage';
import APIPage from './pages/APIPage';
import FeaturesPage from './pages/FeaturesPage';
import './styles/main.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <main className="main">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/demo" element={<DemoPage />} />
            <Route path="/docs" element={<DocsPage />} />
            <Route path="/sdk" element={<SDKPage />} />
            <Route path="/api" element={<APIPage />} />
            <Route path="/features" element={<FeaturesPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
