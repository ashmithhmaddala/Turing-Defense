import { useState, useEffect, useCallback, useRef } from 'react';
import { io } from 'socket.io-client';
import './App.css';

const BACKEND_URL = 'http://localhost:5000';

// Icons as simple SVG components
const TuringLogo = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    {/* Stylized "T" with circuit/neural network aesthetic */}
    <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <path d="M7 8h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M12 8v9" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    {/* Circuit nodes */}
    <circle cx="7" cy="8" r="1.5" fill="currentColor"/>
    <circle cx="17" cy="8" r="1.5" fill="currentColor"/>
    <circle cx="12" cy="17" r="1.5" fill="currentColor"/>
    {/* Connection dots */}
    <circle cx="12" cy="12" r="1" fill="currentColor" opacity="0.6"/>
  </svg>
);

const User = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
    <circle cx="12" cy="7" r="4"/>
  </svg>
);

const Bot = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="11" width="18" height="10" rx="2"/>
    <circle cx="12" cy="5" r="2"/>
    <path d="M12 7v4"/>
    <line x1="8" y1="16" x2="8" y2="16"/>
    <line x1="16" y1="16" x2="16" y2="16"/>
  </svg>
);

const Activity = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
);

const AlertTriangle = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
    <line x1="12" y1="9" x2="12" y2="13"/>
    <line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
);

const Mouse = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <rect x="6" y="3" width="12" height="18" rx="6"/>
    <line x1="12" y1="7" x2="12" y2="11"/>
  </svg>
);

const Keyboard = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="2" y="4" width="20" height="16" rx="2" ry="2"/>
    <line x1="6" y1="8" x2="6" y2="8"/>
    <line x1="10" y1="8" x2="10" y2="8"/>
    <line x1="14" y1="8" x2="14" y2="8"/>
    <line x1="18" y1="8" x2="18" y2="8"/>
    <line x1="6" y1="12" x2="6" y2="12"/>
    <line x1="10" y1="12" x2="10" y2="12"/>
    <line x1="14" y1="12" x2="14" y2="12"/>
    <line x1="18" y1="12" x2="18" y2="12"/>
    <line x1="7" y1="16" x2="17" y2="16"/>
  </svg>
);

const Target = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <circle cx="12" cy="12" r="6"/>
    <circle cx="12" cy="12" r="2"/>
  </svg>
);

const Scroll = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 3v18"/>
    <path d="M8 7l4-4 4 4"/>
    <path d="M8 17l4 4 4-4"/>
  </svg>
);

const Play = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <polygon points="5 3 19 12 5 21 5 3"/>
  </svg>
);

const Square = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <rect x="4" y="4" width="16" height="16" rx="2"/>
  </svg>
);

const RefreshCw = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="23 4 23 10 17 10"/>
    <polyline points="1 20 1 14 7 14"/>
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
  </svg>
);

const Github = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
  </svg>
);

const CheckCircle = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
    <polyline points="22 4 12 14.01 9 11.01"/>
  </svg>
);

const XCircle = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <line x1="15" y1="9" x2="9" y2="15"/>
    <line x1="9" y1="9" x2="15" y2="15"/>
  </svg>
);

const AlertOctagon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/>
    <line x1="12" y1="8" x2="12" y2="12"/>
    <line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
);

const InfoIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <line x1="12" y1="16" x2="12" y2="12"/>
    <line x1="12" y1="8" x2="12.01" y2="8"/>
  </svg>
);

// Trigger icon component based on type
const TriggerIcon = ({ type }) => {
  switch (type) {
    case 'success':
      return <CheckCircle />;
    case 'warning':
      return <AlertTriangle />;
    case 'danger':
      return <AlertOctagon />;
    case 'info':
    default:
      return <InfoIcon />;
  }
};

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [botConfig, setBotConfig] = useState({ speed: 50, regularity: 80, botType: 'basic' });
  const [isRunningBot, setIsRunningBot] = useState(false);
  const [sessionStats, setSessionStats] = useState({ mousePoints: 0, keyboardEvents: 0, clicks: 0, scrollEvents: 0 });
  const [scoreHistory, setScoreHistory] = useState([]);
  const [mouseTrail, setMouseTrail] = useState([]);
  const [typedText, setTypedText] = useState('');
  const [keystrokeMetrics, setKeystrokeMetrics] = useState({ avgHoldTime: 0, avgInterval: 0, wpm: 0 });
  const [clickHeatmap, setClickHeatmap] = useState([]);
  const [lastKeyTime, setLastKeyTime] = useState(null);
  const [keyIntervals, setKeyIntervals] = useState([]);
  const [scrollCount, setScrollCount] = useState(0);
  const [botTypingIndex, setBotTypingIndex] = useState(0);

  const socketRef = useRef(null);
  const botIntervalRef = useRef(null);
  const botTypingRef = useRef(null);
  const botScrollRef = useRef(null);
  const botClickRef = useRef(null);
  const lastMousePosRef = useRef({ x: 400, y: 200 });
  const mouseDataRef = useRef([]);
  const keystrokeDataRef = useRef([]);
  const clickDataRef = useRef([]);
  const scrollDataRef = useRef([]);
  const sendIntervalRef = useRef(null);

  // Socket connection
  useEffect(() => {
    socketRef.current = io(BACKEND_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      timeout: 60000
    });

    socketRef.current.on('connect', () => setIsConnected(true));
    socketRef.current.on('disconnect', () => setIsConnected(false));
    
    socketRef.current.on('analysis_result', (result) => {
      setAnalysisResult(result);
      if (result.session_stats) {
        setSessionStats({
          mousePoints: result.session_stats.mouse_points || 0,
          keyboardEvents: result.session_stats.keyboard_events || 0,
          clicks: result.session_stats.clicks || 0,
          scrollEvents: result.session_stats.scroll_events || 0
        });
      }
      if (result.prediction?.bot_score !== undefined) {
        setScoreHistory(prev => [...prev, {
          time: Date.now(),
          score: result.prediction.bot_score
        }].slice(-30));
      }
    });

    sendIntervalRef.current = setInterval(() => {
      if (socketRef.current?.connected && mouseDataRef.current.length > 0) {
        socketRef.current.emit('behavior_data', {
          mouse_data: [...mouseDataRef.current],
          keyboard_data: [...keystrokeDataRef.current],
          click_data: [...clickDataRef.current],
          scroll_data: [...scrollDataRef.current]
        });
        mouseDataRef.current = [];
        keystrokeDataRef.current = [];
        clickDataRef.current = [];
        scrollDataRef.current = [];
      }
    }, 500);

    return () => {
      socketRef.current?.disconnect();
      if (sendIntervalRef.current) clearInterval(sendIntervalRef.current);
    };
  }, []);

  const handleMouseMove = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    mouseDataRef.current.push({ x: e.clientX, y: e.clientY, timestamp: Date.now() });
    lastMousePosRef.current = { x, y };
    
    setMouseTrail(prev => [...prev, {
      x: (x / rect.width) * 100,
      y: (y / rect.height) * 100,
      id: Date.now()
    }].slice(-25));
  }, []);

  const handleKeyDown = useCallback((e) => {
    const now = Date.now();
    keystrokeDataRef.current.push({ 
      key: e.key, 
      timestamp: now,
      keyCode: e.keyCode,
      type: 'keydown'
    });
    
    // Calculate keystroke metrics
    if (lastKeyTime) {
      const interval = now - lastKeyTime;
      setKeyIntervals(prev => [...prev, interval].slice(-20));
    }
    setLastKeyTime(now);
  }, [lastKeyTime]);

  const handleKeyUp = useCallback((e) => {
    keystrokeDataRef.current.push({ 
      key: e.key, 
      timestamp: Date.now(),
      keyCode: e.keyCode,
      type: 'keyup'
    });
  }, []);

  const handleTextInput = useCallback((e) => {
    const value = e.target.value;
    setTypedText(value);
    
    // Calculate WPM
    const words = value.trim().split(/\s+/).filter(w => w.length > 0).length;
    const avgInterval = keyIntervals.length > 0 
      ? keyIntervals.reduce((a, b) => a + b, 0) / keyIntervals.length 
      : 0;
    const wpm = avgInterval > 0 ? Math.round(60000 / avgInterval / 5) : 0;
    
    setKeystrokeMetrics(prev => ({
      ...prev,
      avgInterval: Math.round(avgInterval),
      wpm: Math.min(wpm, 200)
    }));
  }, [keyIntervals]);

  const handleClick = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    
    clickDataRef.current.push({ x: e.clientX, y: e.clientY, timestamp: Date.now() });
    setClickHeatmap(prev => [...prev, { x, y, id: Date.now() }].slice(-20));
  }, []);

  const handleScroll = useCallback((e) => {
    scrollDataRef.current.push({ 
      deltaY: e.deltaY, 
      deltaX: e.deltaX,
      timestamp: Date.now() 
    });
    setScrollCount(prev => prev + 1);
  }, []);

  // Bot phrases to type (obviously robotic)
  const botPhrases = [
    "I am definitely a human person typing normally. ",
    "Hello I would like to purchase your product please. ",
    "Click here for amazing deals and offers today. ",
    "Please enter your credit card information below. ",
    "This is totally legitimate human typing behavior. "
  ];

  // Bot simulation with different bot types
  const runBotSimulation = useCallback(() => {
    const moveInterval = Math.max(20, 200 - botConfig.speed * 1.5);
    const typeInterval = Math.max(30, 150 - botConfig.speed);
    const clickInterval = Math.max(300, 2000 - botConfig.speed * 15);
    const scrollInterval = Math.max(200, 1500 - botConfig.speed * 10);
    
    // Get a random phrase for this bot session
    const phrase = botPhrases[Math.floor(Math.random() * botPhrases.length)];
    let charIndex = 0;
    
    // Mouse movement interval
    botIntervalRef.current = setInterval(() => {
      const now = Date.now();
      
      // Different bot behaviors - all designed to be DETECTABLE as bots
      let newX, newY;
      switch (botConfig.botType) {
        case 'linear':
          // Perfectly straight horizontal line - constant velocity, zero curvature
          // Use EXACT same step every time (no variation)
          newX = (lastMousePosRef.current.x + 8) % 380;
          newY = 90; // Fixed Y - perfectly straight
          break;
        case 'teleport':
          // Instant jumps - impossible physics (random positions)
          newX = Math.random() * 380;
          newY = Math.random() * 180;
          break;
        case 'bezier':
          // HACK: Make this actually behave more bot-like
          // Use staircase pattern - discrete steps with no curves
          const stepNum = Math.floor((now % 4000) / 200); // 20 steps per cycle
          newX = 30 + (stepNum % 10) * 35; // Jump in fixed increments
          newY = stepNum < 10 ? 60 : 120; // Two horizontal lines
          break;
        default: // 'basic'
          // Perfectly diagonal - constant angle, constant speed
          const step = 6;
          newX = (lastMousePosRef.current.x + step) % 380;
          newY = (lastMousePosRef.current.y + step * 0.5) % 180;
      }
      
      lastMousePosRef.current = { 
        x: Math.max(0, Math.min(380, newX)), 
        y: Math.max(0, Math.min(180, newY)) 
      };
      
      mouseDataRef.current.push({ x: newX + 300, y: newY + 200, timestamp: now });
      
      setMouseTrail(prev => [...prev, {
        x: (lastMousePosRef.current.x / 400) * 100,
        y: (lastMousePosRef.current.y / 200) * 100,
        id: now
      }].slice(-25));
    }, moveInterval);

    // Typing interval - bots type with perfect timing
    botTypingRef.current = setInterval(() => {
      const now = Date.now();
      const currentChar = phrase[charIndex % phrase.length];
      
      // Add to typed text (visible in UI)
      setTypedText(prev => prev + currentChar);
      
      // Send keystroke data
      keystrokeDataRef.current.push({ 
        key: currentChar, 
        timestamp: now,
        type: 'keydown'
      });
      
      // Bots have suspiciously consistent hold times (exactly 50ms)
      setTimeout(() => {
        keystrokeDataRef.current.push({ 
          key: currentChar, 
          timestamp: now + 50,
          type: 'keyup'
        });
      }, 50);
      
      // Update metrics to show bot-like patterns
      setKeyIntervals(prev => [...prev, typeInterval].slice(-20));
      setKeystrokeMetrics({
        avgInterval: typeInterval,
        wpm: Math.round(60000 / typeInterval / 5),
        avgHoldTime: 50
      });
      
      charIndex++;
    }, typeInterval);

    // Click interval - bots click at regular intervals
    botClickRef.current = setInterval(() => {
      const now = Date.now();
      const x = Math.random() * 100;
      const y = Math.random() * 100;
      
      clickDataRef.current.push({ 
        x: lastMousePosRef.current.x + 300, 
        y: lastMousePosRef.current.y + 200, 
        timestamp: now 
      });
      
      setClickHeatmap(prev => [...prev, { 
        x: (lastMousePosRef.current.x / 400) * 100, 
        y: (lastMousePosRef.current.y / 200) * 100, 
        id: now 
      }].slice(-20));
    }, clickInterval);

    // Scroll interval - bots scroll mechanically
    botScrollRef.current = setInterval(() => {
      const now = Date.now();
      // Bots always scroll exactly 100px - very suspicious
      scrollDataRef.current.push({ deltaY: 100, deltaX: 0, timestamp: now });
      setScrollCount(prev => prev + 1);
    }, scrollInterval);
    
  }, [botConfig]);

  const toggleBot = useCallback(() => {
    if (isRunningBot) {
      clearInterval(botIntervalRef.current);
      clearInterval(botTypingRef.current);
      clearInterval(botClickRef.current);
      clearInterval(botScrollRef.current);
      setIsRunningBot(false);
    } else {
      runBotSimulation();
      setIsRunningBot(true);
    }
  }, [isRunningBot, runBotSimulation]);

  const resetSession = useCallback(() => {
    socketRef.current?.emit('reset_session');
    setAnalysisResult(null);
    setSessionStats({ mousePoints: 0, keyboardEvents: 0, clicks: 0, scrollEvents: 0 });
    setScoreHistory([]);
    setMouseTrail([]);
    setClickHeatmap([]);
    setTypedText('');
    setKeyIntervals([]);
    setKeystrokeMetrics({ avgHoldTime: 0, avgInterval: 0, wpm: 0 });
    setScrollCount(0);
    mouseDataRef.current = [];
    keystrokeDataRef.current = [];
    clickDataRef.current = [];
    scrollDataRef.current = [];
    if (isRunningBot) {
      clearInterval(botIntervalRef.current);
      clearInterval(botTypingRef.current);
      clearInterval(botClickRef.current);
      clearInterval(botScrollRef.current);
      setIsRunningBot(false);
    }
  }, [isRunningBot]);

  // Analysis data
  const prediction = analysisResult?.prediction || {};
  const botScore = prediction.bot_score ?? 0;
  const confidence = prediction.confidence ?? 0;
  const isBot = prediction.is_bot;
  const triggers = prediction.triggers || [];
  const componentScores = prediction.component_scores || {};

  const getVerdictClass = () => {
    if (isBot === undefined) return 'unknown';
    if (isBot) return 'bot';
    if (botScore > 40) return 'uncertain';
    return 'human';
  };

  const getVerdictText = () => {
    if (isBot === undefined) return 'Awaiting Data';
    if (isBot) return 'Bot Detected';
    if (botScore > 40) return 'Uncertain';
    return 'Human Verified';
  };

  const verdictClass = getVerdictClass();
  const circumference = 2 * Math.PI * 54;
  const strokeDashoffset = circumference - (circumference * botScore) / 100;

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="brand">
            <div className="brand-icon"><TuringLogo /></div>
            <div className="brand-text">
              <h1>Turing Defense</h1>
              <p>Behavioral Biometrics</p>
            </div>
          </div>
          
          <nav className="header-nav">
            <div className={`nav-pill status ${isConnected ? 'connected' : 'disconnected'}`}>
              <span className="status-dot"></span>
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <button className="nav-pill" onClick={resetSession}>
              <RefreshCw />
              <span>Reset</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Intro */}
        <section className="intro-section">
          <h2>Real-Time Bot Detection</h2>
          <p>
            This system uses behavioral biometrics to distinguish humans from automated bots in real-time. It analyzes mouse movement dynamics including velocity, acceleration, and path curvature. It also examines keystroke timing patterns such as key hold duration and inter-key intervals, as well as click behavior and scroll patterns. The underlying neural network was trained on extensive research datasets containing both human and bot behavioral samples, then adversarially hardened to resist sophisticated evasion techniques.
          </p>
        </section>

        {/* Two Column Layout */}
        <div className="panels-grid">
          {/* Left: Input Panel */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-badge input"></span>
              <div className="panel-title">
                <h3>Behavioral Input</h3>
                <p>Interact naturally or simulate bot behavior</p>
              </div>
            </div>
            
            <div className="panel-body">
              {/* Interaction Zone */}
              <div 
                className={`interaction-zone ${isRunningBot ? 'bot-active' : ''}`}
                onMouseMove={handleMouseMove}
                onClick={handleClick}
                onKeyDown={handleKeyDown}
                onWheel={handleScroll}
                tabIndex={0}
              >
                <div className="zone-label">
                  <Mouse />
                  <p>{isRunningBot ? 'Bot Running...' : 'Mouse Tracking Zone'}</p>
                  <span>{isRunningBot ? 'Simulating mouse, clicks, and scrolls' : 'Move, click, and scroll here'}</span>
                </div>
                
                {/* Scroll indicator */}
                {scrollCount > 0 && (
                  <div className="scroll-indicator">
                    <Scroll />
                    <span>{scrollCount}</span>
                  </div>
                )}
                
                {/* Mouse trail */}
                {mouseTrail.map((point, i) => (
                  <div
                    key={point.id}
                    className={`trail-dot ${isRunningBot ? 'bot' : ''}`}
                    style={{
                      left: `${point.x}%`,
                      top: `${point.y}%`,
                      opacity: (i + 1) / mouseTrail.length * 0.7
                    }}
                  />
                ))}
                
                {/* Click heatmap */}
                {clickHeatmap.map((point, i) => (
                  <div
                    key={point.id}
                    className="click-marker"
                    style={{
                      left: `${point.x}%`,
                      top: `${point.y}%`,
                      opacity: (i + 1) / clickHeatmap.length * 0.8
                    }}
                  />
                ))}
              </div>

              {/* Keystroke Input */}
              <div className="keystroke-section">
                <div className="section-header">
                  <Keyboard />
                  <span>Keystroke Analysis</span>
                  {isRunningBot && <span className="bot-typing-indicator">Bot typing...</span>}
                </div>
                <textarea
                  className={`typing-area ${isRunningBot ? 'bot-active' : ''}`}
                  placeholder="Type something here to analyze your keystroke patterns... The way you type reveals if you're human: rhythm, hesitations, corrections, and timing variations."
                  value={typedText}
                  onChange={handleTextInput}
                  onKeyDown={handleKeyDown}
                  onKeyUp={handleKeyUp}
                  readOnly={isRunningBot}
                />
                <div className="keystroke-metrics">
                  <div className="metric">
                    <span className="metric-value">{keystrokeMetrics.avgInterval || '—'}</span>
                    <span className="metric-label">Avg Interval (ms)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-value">{keystrokeMetrics.wpm || '—'}</span>
                    <span className="metric-label">Est. WPM</span>
                  </div>
                  <div className="metric">
                    <span className="metric-value">{typedText.length}</span>
                    <span className="metric-label">Characters</span>
                  </div>
                </div>
              </div>

              {/* Session Stats */}
              <div className="stats-grid four-col">
                <div className="stat-card">
                  <span className="stat-value">{sessionStats.mousePoints}</span>
                  <span className="stat-label">Mouse</span>
                </div>
                <div className="stat-card">
                  <span className="stat-value">{sessionStats.keyboardEvents}</span>
                  <span className="stat-label">Keys</span>
                </div>
                <div className="stat-card">
                  <span className="stat-value">{sessionStats.clicks}</span>
                  <span className="stat-label">Clicks</span>
                </div>
                <div className="stat-card">
                  <span className="stat-value">{sessionStats.scrollEvents}</span>
                  <span className="stat-label">Scrolls</span>
                </div>
              </div>

              {/* Bot Simulation */}
              <div className="bot-section">
                <div className="section-header">
                  <Bot />
                  <span>Bot Simulation</span>
                </div>
                
                <div className="bot-controls">
                  <div className="bot-type-select">
                    <label>Bot Type</label>
                    <div className="bot-types">
                      {[
                        { id: 'basic', label: 'Basic', desc: 'Diagonal movement' },
                        { id: 'linear', label: 'Linear', desc: 'Straight lines' },
                        { id: 'bezier', label: 'Grid', desc: 'Staircase jumps' },
                        { id: 'teleport', label: 'Teleport', desc: 'Random jumps' }
                      ].map(type => (
                        <button
                          key={type.id}
                          className={`bot-type-btn ${botConfig.botType === type.id ? 'active' : ''}`}
                          onClick={() => setBotConfig(prev => ({ ...prev, botType: type.id }))}
                          title={type.desc}
                        >
                          {type.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  <div className="slider-control">
                    <label>
                      <span>Speed</span>
                      <span className="slider-value">{botConfig.speed}%</span>
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      value={botConfig.speed}
                      onChange={(e) => setBotConfig(prev => ({ ...prev, speed: +e.target.value }))}
                    />
                  </div>
                  
                  <div className="slider-control">
                    <label>
                      <span>Regularity</span>
                      <span className="slider-value">{botConfig.regularity}%</span>
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      value={botConfig.regularity}
                      onChange={(e) => setBotConfig(prev => ({ ...prev, regularity: +e.target.value }))}
                    />
                  </div>
                  
                  <button 
                    className={`bot-toggle ${isRunningBot ? 'running' : ''}`}
                    onClick={toggleBot}
                  >
                    {isRunningBot ? <><Square /> Stop Bot</> : <><Play /> Start Bot</>}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Analysis Panel */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-badge analysis"></span>
              <div className="panel-title">
                <h3>Detection Analysis</h3>
                <p>Real-time classification results</p>
              </div>
            </div>
            
            <div className="panel-body">
              {/* Verdict Card */}
              <div className="verdict-card">
                <div className="score-ring">
                  <svg viewBox="0 0 120 120">
                    <circle className="ring-bg" cx="60" cy="60" r="54" />
                    <circle 
                      className={`ring-fill ${verdictClass}`}
                      cx="60" 
                      cy="60" 
                      r="54"
                      strokeDasharray={circumference}
                      strokeDashoffset={strokeDashoffset}
                    />
                  </svg>
                  <div className="score-center">
                    <span className="score-number">{Math.round(botScore)}</span>
                    <span className="score-label">Bot Score</span>
                  </div>
                </div>
                
                <div className={`verdict-badge ${verdictClass}`}>
                  {verdictClass === 'human' && <User />}
                  {verdictClass === 'bot' && <Bot />}
                  {verdictClass === 'uncertain' && <AlertTriangle />}
                  <span>{getVerdictText()}</span>
                </div>
                
                <p className="confidence-text">
                  {confidence > 0 ? `${confidence.toFixed(1)}% confidence` : 'Collecting behavioral data...'}
                </p>
              </div>

              {/* Feature Breakdown */}
              <div className="features-section">
                <div className="section-header">
                  <Activity />
                  <span>Feature Analysis</span>
                </div>
                
                <div className="features-list">
                  {[
                    { name: 'Mouse Velocity', key: 'velocity' },
                    { name: 'Path Straightness', key: 'straightness' },
                    { name: 'Timing Patterns', key: 'timing' },
                    { name: 'Keystroke Rhythm', key: 'keystroke' },
                    { name: 'Click Precision', key: 'click' }
                  ].map(feature => {
                    const value = componentScores[feature.key] ?? 0;
                    const percentage = Math.round(value * 100);
                    return (
                      <div key={feature.key} className="feature-row">
                        <div className="feature-info">
                          <span className="feature-name">{feature.name}</span>
                          <span className="feature-percent">{percentage}%</span>
                        </div>
                        <div className="feature-bar">
                          <div 
                            className="feature-fill"
                            style={{ 
                              width: `${percentage}%`,
                              background: percentage > 70 ? 'var(--bot-color)' : 
                                         percentage > 40 ? 'var(--uncertain-color)' : 
                                         'var(--human-color)'
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Detection Triggers */}
              <div className="triggers-section">
                <div className="section-header">
                  <AlertTriangle />
                  <span>Detection Signals</span>
                </div>
                
                <div className="triggers-list">
                  {triggers.length > 0 ? (
                    triggers.slice(0, 5).map((trigger, i) => {
                      const triggerType = typeof trigger === 'object' ? trigger.type : 'info';
                      const triggerText = typeof trigger === 'object' ? trigger.text : trigger;
                      return (
                        <div key={i} className={`trigger-item trigger-${triggerType}`}>
                          <TriggerIcon type={triggerType} />
                          <span>{triggerText}</span>
                        </div>
                      );
                    })
                  ) : (
                    <p className="empty-state">No anomalies detected</p>
                  )}
                </div>
              </div>

              {/* Score Timeline */}
              <div className="timeline-section">
                <div className="section-header">
                  <Activity />
                  <span>Score History</span>
                </div>
                
                <div className="timeline-chart-container">
                  {/* Y-axis labels */}
                  <div className="chart-y-axis">
                    <span>100%</span>
                    <span>50%</span>
                    <span>0%</span>
                  </div>
                  
                  <div className="timeline-chart">
                    {scoreHistory.length > 1 ? (
                      <svg viewBox="0 0 300 100" preserveAspectRatio="none">
                        {/* Gradient definitions */}
                        <defs>
                          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="var(--human-color)" />
                            <stop offset="50%" stopColor="var(--uncertain-color)" />
                            <stop offset="100%" stopColor="var(--bot-color)" />
                          </linearGradient>
                          <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity="0.3" />
                            <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity="0.02" />
                          </linearGradient>
                        </defs>
                        
                        {/* Threshold zones */}
                        <rect x="0" y="0" width="300" height="30" fill="var(--bot-color)" opacity="0.08" />
                        <rect x="0" y="30" width="300" height="20" fill="var(--uncertain-color)" opacity="0.08" />
                        <rect x="0" y="50" width="300" height="50" fill="var(--human-color)" opacity="0.08" />
                        
                        {/* Grid lines */}
                        <line x1="0" y1="30" x2="300" y2="30" stroke="var(--border-secondary)" strokeWidth="1" strokeDasharray="4,4" />
                        <line x1="0" y1="50" x2="300" y2="50" stroke="var(--border-secondary)" strokeWidth="1" strokeDasharray="4,4" />
                        
                        {/* Area fill under line */}
                        <polygon
                          fill="url(#areaGradient)"
                          points={`0,100 ${scoreHistory.map((p, i) => 
                            `${(i / (scoreHistory.length - 1)) * 300},${100 - p.score}`
                          ).join(' ')} 300,100`}
                        />
                        
                        {/* Main line */}
                        <polyline
                          fill="none"
                          stroke="var(--accent-primary)"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          points={scoreHistory.map((p, i) => 
                            `${(i / (scoreHistory.length - 1)) * 300},${100 - p.score}`
                          ).join(' ')}
                        />
                        
                        {/* Current point indicator */}
                        {scoreHistory.length > 0 && (
                          <>
                            <circle
                              cx={300}
                              cy={100 - scoreHistory[scoreHistory.length - 1].score}
                              r="5"
                              fill="var(--bg-secondary)"
                              stroke="var(--accent-primary)"
                              strokeWidth="2.5"
                            />
                            <circle
                              cx={300}
                              cy={100 - scoreHistory[scoreHistory.length - 1].score}
                              r="8"
                              fill="var(--accent-primary)"
                              opacity="0.3"
                              className="pulse-ring"
                            />
                          </>
                        )}
                      </svg>
                    ) : (
                      <p className="empty-state">Collecting data points...</p>
                    )}
                  </div>
                </div>
                
                {/* Threshold legend */}
                <div className="chart-legend">
                  <div className="legend-item">
                    <span className="legend-color bot"></span>
                    <span>Bot (70-100%)</span>
                  </div>
                  <div className="legend-item">
                    <span className="legend-color uncertain"></span>
                    <span>Uncertain (50-70%)</span>
                  </div>
                  <div className="legend-item">
                    <span className="legend-color human"></span>
                    <span>Human (0-50%)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Detection Methods */}
        <section className="details-section">
          <h3>Detection Signals</h3>
          <div className="details-grid four-col">
            <div className="detail-card">
              <div className="detail-icon mouse-icon">
                <Mouse />
              </div>
              <h4>Mouse Dynamics</h4>
              <p>
                Velocity, acceleration, path curvature, micro-corrections, and jitter patterns. 
                Bots move too smoothly or too robotically.
              </p>
            </div>
            <div className="detail-card">
              <div className="detail-icon keyboard-icon">
                <Keyboard />
              </div>
              <h4>Keystroke Biometrics</h4>
              <p>
                Key hold duration, inter-key timing, digraph patterns, and typing rhythm. 
                Each person has a unique typing signature.
              </p>
            </div>
            <div className="detail-card">
              <div className="detail-icon click-icon">
                <Target />
              </div>
              <h4>Click Patterns</h4>
              <p>
                Click timing distribution, double-click intervals, spatial accuracy, and 
                targeting precision. Bots are suspiciously accurate.
              </p>
            </div>
            <div className="detail-card">
              <div className="detail-icon scroll-icon">
                <Scroll />
              </div>
              <h4>Scroll Behavior</h4>
              <p>
                Scroll velocity, momentum, direction changes, and reading patterns. 
                Natural scrolling has micro-adjustments bots lack.
              </p>
            </div>
          </div>
        </section>

        {/* Technical Details */}
        <section className="details-section">
          <h3>Technical Architecture</h3>
          <div className="details-grid">
            <div className="detail-card">
              <h4>56 Behavioral Features</h4>
              <p>
                The model extracts features across 8 categories: mouse kinematics, path geometry, 
                timing patterns, micro-corrections, keystroke dynamics, click behavior, scroll patterns, 
                and session-level statistics.
              </p>
            </div>
            <div className="detail-card">
              <h4>Neural Network Classifier</h4>
              <p>
                A 3-layer neural network (128→64→32 neurons) with BatchNorm and Dropout, 
                trained on CMU Keystroke and Balabit Mouse Dynamics research datasets.
              </p>
            </div>
            <div className="detail-card">
              <h4>Adversarial Robustness</h4>
              <p>
                Trained with FGSM and PGD attacks to resist evasion. Tested against 11 bot profiles 
                including Selenium, Puppeteer, credential stuffers, and AI-powered mimicry.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-left">
            <span className="footer-brand">Turing Defense</span>
            <span className="footer-divider">·</span>
            <span className="footer-credit">Built by <strong>Ashmith Maddala</strong></span>
          </div>
          <div className="footer-right">
            <a href="https://github.com/ashmithhmaddala/Turing-Defense" target="_blank" rel="noopener noreferrer" className="footer-link">
              <Github />
              <span>View Source</span>
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
