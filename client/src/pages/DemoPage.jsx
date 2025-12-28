import { useState, useEffect, useCallback, useRef } from 'react';
import { io } from 'socket.io-client';
import { Link } from 'react-router-dom';
import { 
  User, Bot, Activity, AlertTriangle, Mouse, Keyboard, 
  Target, Scroll, Play, Square, CheckCircle, AlertOctagon, InfoIcon,
  Zap, Eye, Shield, ArrowRight, Gauge, Ruler, Timer, Crosshair,
  TrendingUp, Fingerprint, Layers, Clock, BarChart
} from '../components/Icons';

const BACKEND_URL = 'http://localhost:5000';

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

const RefreshCw = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="23 4 23 10 17 10"/>
    <polyline points="1 20 1 14 7 14"/>
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
  </svg>
);

const Settings = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="3"/>
    <path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
  </svg>
);

export default function DemoPage() {
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
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [sessionDuration, setSessionDuration] = useState(0);
  const [mouseVelocity, setMouseVelocity] = useState(0);
  const [pathStraightness, setPathStraightness] = useState(0);
  const [clickFrequency, setClickFrequency] = useState(0);
  const [avgScrollDelta, setAvgScrollDelta] = useState(0);
  const lastMouseTimeRef = useRef(null);
  const velocitiesRef = useRef([]);

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
  const sessionTimerRef = useRef(null);

  // Session timer
  useEffect(() => {
    sessionTimerRef.current = setInterval(() => {
      setSessionDuration(prev => prev + 1);
    }, 1000);
    return () => clearInterval(sessionTimerRef.current);
  }, []);

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
    const target = e.currentTarget;
    const rect = target.getBoundingClientRect();
    
    // clientWidth/clientHeight give inner dimensions (excluding border)
    const contentWidth = target.clientWidth;
    const contentHeight = target.clientHeight;
    
    // Calculate border offset dynamically
    const borderX = (rect.width - contentWidth) / 2;
    const borderY = (rect.height - contentHeight) / 2;
    
    // Position relative to inner content area
    const x = e.clientX - rect.left - borderX;
    const y = e.clientY - rect.top - borderY;
    const now = Date.now();
    
    // Clamp values to prevent going outside bounds
    const clampedX = Math.max(0, Math.min(x, contentWidth));
    const clampedY = Math.max(0, Math.min(y, contentHeight));
    
    // Calculate velocity
    if (lastMouseTimeRef.current && lastMousePosRef.current) {
      const dt = now - lastMouseTimeRef.current;
      const dx = clampedX - lastMousePosRef.current.x;
      const dy = clampedY - lastMousePosRef.current.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const velocity = dt > 0 ? (distance / dt) * 1000 : 0; // pixels per second
      
      velocitiesRef.current = [...velocitiesRef.current, velocity].slice(-20);
      const avgVelocity = velocitiesRef.current.reduce((a, b) => a + b, 0) / velocitiesRef.current.length;
      setMouseVelocity(Math.round(avgVelocity));
      
      // Calculate path straightness (simplified)
      if (mouseTrail.length >= 3) {
        const recent = mouseTrail.slice(-5);
        let totalAngleChange = 0;
        for (let i = 2; i < recent.length; i++) {
          const dx1 = recent[i-1].x - recent[i-2].x;
          const dy1 = recent[i-1].y - recent[i-2].y;
          const dx2 = recent[i].x - recent[i-1].x;
          const dy2 = recent[i].y - recent[i-1].y;
          const angle1 = Math.atan2(dy1, dx1);
          const angle2 = Math.atan2(dy2, dx2);
          totalAngleChange += Math.abs(angle2 - angle1);
        }
        const straightness = Math.max(0, 100 - (totalAngleChange * 20));
        setPathStraightness(Math.round(straightness));
      }
    }
    
    lastMouseTimeRef.current = now;
    mouseDataRef.current.push({ x: e.clientX, y: e.clientY, timestamp: now });
    lastMousePosRef.current = { x: clampedX, y: clampedY };
    
    setMouseTrail(prev => [...prev, {
      x: (clampedX / contentWidth) * 100,
      y: (clampedY / contentHeight) * 100,
      id: now
    }].slice(-30));
  }, [mouseTrail]);

  const handleKeyDown = useCallback((e) => {
    const now = Date.now();
    keystrokeDataRef.current.push({ 
      key: e.key, 
      timestamp: now,
      keyCode: e.keyCode,
      type: 'keydown'
    });
    
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
    const target = e.currentTarget;
    const rect = target.getBoundingClientRect();
    
    // clientWidth/clientHeight give inner dimensions (excluding border)
    const contentWidth = target.clientWidth;
    const contentHeight = target.clientHeight;
    
    // Calculate border offset dynamically
    const borderX = (rect.width - contentWidth) / 2;
    const borderY = (rect.height - contentHeight) / 2;
    
    // Position relative to inner content area
    const x = e.clientX - rect.left - borderX;
    const y = e.clientY - rect.top - borderY;
    const now = Date.now();
    
    const xPercent = (Math.max(0, Math.min(x, contentWidth)) / contentWidth) * 100;
    const yPercent = (Math.max(0, Math.min(y, contentHeight)) / contentHeight) * 100;
    
    clickDataRef.current.push({ x: e.clientX, y: e.clientY, timestamp: now });
    setClickHeatmap(prev => {
      const newHeatmap = [...prev, { x: xPercent, y: yPercent, id: now }].slice(-20);
      // Calculate click frequency (clicks per minute)
      if (newHeatmap.length >= 2) {
        const timeSpan = (now - newHeatmap[0].id) / 1000; // seconds
        if (timeSpan > 0) {
          setClickFrequency(Math.round((newHeatmap.length / timeSpan) * 60));
        }
      }
      return newHeatmap;
    });
  }, []);

  const handleScroll = useCallback((e) => {
    scrollDataRef.current.push({ 
      deltaY: e.deltaY, 
      deltaX: e.deltaX,
      timestamp: Date.now() 
    });
    setScrollCount(prev => prev + 1);
    
    // Calculate average scroll delta
    const scrolls = scrollDataRef.current.slice(-20);
    const avgDelta = scrolls.reduce((a, b) => a + Math.abs(b.deltaY), 0) / scrolls.length;
    setAvgScrollDelta(Math.round(avgDelta));
  }, []);

  const botPhrases = [
    "I am definitely a human person typing normally. ",
    "Hello I would like to purchase your product please. ",
    "Click here for amazing deals and offers today. ",
    "Please enter your credit card information below. ",
    "This is totally legitimate human typing behavior. "
  ];

  const runBotSimulation = useCallback(() => {
    const moveInterval = Math.max(20, 200 - botConfig.speed * 1.5);
    const typeInterval = Math.max(30, 150 - botConfig.speed);
    const clickInterval = Math.max(300, 2000 - botConfig.speed * 15);
    const scrollInterval = Math.max(200, 1500 - botConfig.speed * 10);
    
    const phrase = botPhrases[Math.floor(Math.random() * botPhrases.length)];
    let charIndex = 0;
    
    botIntervalRef.current = setInterval(() => {
      const now = Date.now();
      let newX, newY;
      
      switch (botConfig.botType) {
        case 'linear':
          newX = (lastMousePosRef.current.x + 8) % 380;
          newY = 90;
          break;
        case 'teleport':
          newX = Math.random() * 380;
          newY = Math.random() * 180;
          break;
        case 'bezier':
          const stepNum = Math.floor((now % 4000) / 200);
          newX = 30 + (stepNum % 10) * 35;
          newY = stepNum < 10 ? 60 : 120;
          break;
        default:
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
      }].slice(-30));
    }, moveInterval);

    botTypingRef.current = setInterval(() => {
      const now = Date.now();
      const currentChar = phrase[charIndex % phrase.length];
      
      setTypedText(prev => prev + currentChar);
      
      keystrokeDataRef.current.push({ 
        key: currentChar, 
        timestamp: now,
        type: 'keydown'
      });
      
      setTimeout(() => {
        keystrokeDataRef.current.push({ 
          key: currentChar, 
          timestamp: now + 50,
          type: 'keyup'
        });
      }, 50);
      
      setKeyIntervals(prev => [...prev, typeInterval].slice(-20));
      setKeystrokeMetrics({
        avgInterval: typeInterval,
        wpm: Math.round(60000 / typeInterval / 5),
        avgHoldTime: 50
      });
      
      charIndex++;
    }, typeInterval);

    botClickRef.current = setInterval(() => {
      const now = Date.now();
      
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

    botScrollRef.current = setInterval(() => {
      const now = Date.now();
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
    setSessionDuration(0);
    setMouseVelocity(0);
    setPathStraightness(0);
    setClickFrequency(0);
    setAvgScrollDelta(0);
    velocitiesRef.current = [];
    lastMouseTimeRef.current = null;
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

  const formatDuration = (seconds) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const totalEvents = sessionStats.mousePoints + sessionStats.keyboardEvents + sessionStats.clicks + sessionStats.scrollEvents;

  const verdictClass = getVerdictClass();
  const circumference = 2 * Math.PI * 54;
  const strokeDashoffset = circumference - (circumference * botScore) / 100;

  return (
    <div className="demo-page">
      {/* Hero Banner */}
      <section className="demo-hero">
        <div className="demo-hero-content">
          <h1>Interactive Bot Detection Demo</h1>
          <p>Experience real-time behavioral analysis. Move your mouse, type, click, and scroll to see how our neural network distinguishes humans from bots.</p>
          <div className="demo-hero-badges">
            <div className="demo-badge">
              <Zap size={16} />
              <span>Real-time Analysis</span>
            </div>
            <div className="demo-badge">
              <Eye size={16} />
              <span>56 Features</span>
            </div>
            <div className="demo-badge">
              <Shield size={16} />
              <span>91.8% Accuracy</span>
            </div>
          </div>
        </div>
      </section>

      {/* Status Bar */}
      <div className="demo-status-bar">
        <div className="demo-status-content">
          <div className="status-left">
            <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
              <span className="status-indicator"></span>
              <span>{isConnected ? 'Connected to Server' : 'Disconnected'}</span>
            </div>
            <div className="session-timer">
              <span className="timer-label">Session:</span>
              <span className="timer-value">{formatDuration(sessionDuration)}</span>
            </div>
            <div className="total-events">
              <span className="events-label">Events:</span>
              <span className="events-value">{totalEvents.toLocaleString()}</span>
            </div>
          </div>
          <div className="status-right">
            <button className="btn btn-secondary btn-sm" onClick={resetSession}>
              <RefreshCw />
              <span>Reset</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Demo Grid */}
      <div className="demo-main">
        <div className="demo-container">
          <div className="demo-layout">
            {/* Left Column - Input */}
            <div className="demo-column input-column">
              <div className="demo-card">
                <div className="card-header">
                  <div className="card-title-group">
                    <div className="card-icon">
                      <Mouse size={20} />
                    </div>
                    <div>
                      <h3>Mouse & Click Tracking</h3>
                      <p>Move and click inside the zone</p>
                    </div>
                  </div>
                  <div className="card-stats">
                    <span className="mini-stat">{sessionStats.mousePoints} moves</span>
                    <span className="mini-stat">{sessionStats.clicks} clicks</span>
                  </div>
                </div>
                
                <div 
                  className={`tracking-zone ${isRunningBot ? 'bot-active' : ''}`}
                  onMouseMove={handleMouseMove}
                  onClick={handleClick}
                  onWheel={handleScroll}
                  tabIndex={0}
                >
                  {!isRunningBot && mouseTrail.length === 0 && (
                    <div className="zone-placeholder">
                      <Mouse size={48} />
                      <p>Move your mouse here</p>
                      <span>Your movement patterns will be analyzed in real-time</span>
                    </div>
                  )}
                  
                  {isRunningBot && (
                    <div className="bot-indicator">
                      <Bot size={24} />
                      <span>Bot Simulation Active</span>
                    </div>
                  )}
                  
                  {scrollCount > 0 && (
                    <div className="scroll-badge">
                      <Scroll size={14} />
                      <span>{scrollCount} scrolls</span>
                    </div>
                  )}
                  
                  {/* Mouse trail visualization */}
                  <svg className="trail-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                    {mouseTrail.length > 1 && (
                      <polyline
                        className={`trail-line ${isRunningBot ? 'bot' : 'human'}`}
                        points={mouseTrail.map(p => `${p.x},${p.y}`).join(' ')}
                        fill="none"
                      />
                    )}
                    {mouseTrail.length > 0 && (
                      <circle
                        className={`trail-head ${isRunningBot ? 'bot' : 'human'}`}
                        cx={mouseTrail[mouseTrail.length - 1].x}
                        cy={mouseTrail[mouseTrail.length - 1].y}
                        r="2"
                      />
                    )}
                  </svg>
                  
                  {/* Click markers */}
                  {clickHeatmap.map((point, i) => (
                    <div
                      key={point.id}
                      className="click-ripple"
                      style={{
                        left: `${point.x}%`,
                        top: `${point.y}%`,
                        opacity: (i + 1) / clickHeatmap.length
                      }}
                    />
                  ))}
                </div>
              </div>

              <div className="demo-card">
                <div className="card-header">
                  <div className="card-title-group">
                    <div className="card-icon">
                      <Keyboard size={20} />
                    </div>
                    <div>
                      <h3>Keystroke Dynamics</h3>
                      <p>Type to analyze your rhythm</p>
                    </div>
                  </div>
                  <div className="card-stats">
                    <span className="mini-stat">{sessionStats.keyboardEvents} keys</span>
                  </div>
                </div>
                
                <textarea
                  className={`typing-input ${isRunningBot ? 'bot-active' : ''}`}
                  placeholder="Start typing here... We'll analyze your keystroke timing, rhythm, and patterns."
                  value={typedText}
                  onChange={handleTextInput}
                  onKeyDown={handleKeyDown}
                  onKeyUp={handleKeyUp}
                  readOnly={isRunningBot}
                />
                
                <div className="keystroke-stats">
                  <div className="keystroke-stat">
                    <span className="ks-value">{keystrokeMetrics.avgInterval || '—'}</span>
                    <span className="ks-label">Avg Interval (ms)</span>
                  </div>
                  <div className="keystroke-stat">
                    <span className="ks-value">{keystrokeMetrics.wpm || '—'}</span>
                    <span className="ks-label">Words/Min</span>
                  </div>
                  <div className="keystroke-stat">
                    <span className="ks-value">{typedText.length}</span>
                    <span className="ks-label">Characters</span>
                  </div>
                </div>
              </div>

              {/* Real-time Metrics Panel */}
              <div className="demo-card metrics-card">
                <div className="card-header">
                  <div className="card-title-group">
                    <div className="card-icon">
                      <BarChart size={20} />
                    </div>
                    <div>
                      <h3>Real-time Metrics</h3>
                      <p>Live behavioral measurements</p>
                    </div>
                  </div>
                </div>
                
                <div className="metrics-grid">
                  <div className="metric-box">
                    <div className="metric-header">
                      <Gauge size={16} />
                      <span>Mouse Velocity</span>
                    </div>
                    <div className="metric-value">{mouseVelocity}<span className="metric-unit">px/s</span></div>
                    <div className="metric-bar">
                      <div className="metric-fill" style={{ width: `${Math.min(100, mouseVelocity / 20)}%` }}></div>
                    </div>
                  </div>
                  
                  <div className="metric-box">
                    <div className="metric-header">
                      <Ruler size={16} />
                      <span>Path Straightness</span>
                    </div>
                    <div className="metric-value">{pathStraightness}<span className="metric-unit">%</span></div>
                    <div className="metric-bar">
                      <div className="metric-fill" style={{ width: `${pathStraightness}%` }}></div>
                    </div>
                  </div>
                  
                  <div className="metric-box">
                    <div className="metric-header">
                      <Crosshair size={16} />
                      <span>Click Frequency</span>
                    </div>
                    <div className="metric-value">{clickFrequency}<span className="metric-unit">/min</span></div>
                    <div className="metric-bar">
                      <div className="metric-fill" style={{ width: `${Math.min(100, clickFrequency / 1.2)}%` }}></div>
                    </div>
                  </div>
                  
                  <div className="metric-box">
                    <div className="metric-header">
                      <Scroll size={16} />
                      <span>Scroll Intensity</span>
                    </div>
                    <div className="metric-value">{avgScrollDelta}<span className="metric-unit">delta</span></div>
                    <div className="metric-bar">
                      <div className="metric-fill" style={{ width: `${Math.min(100, avgScrollDelta / 2)}%` }}></div>
                    </div>
                  </div>
                  
                  <div className="metric-box">
                    <div className="metric-header">
                      <Timer size={16} />
                      <span>Key Rhythm</span>
                    </div>
                    <div className="metric-value">{keystrokeMetrics.avgInterval || 0}<span className="metric-unit">ms</span></div>
                    <div className="metric-bar">
                      <div className="metric-fill" style={{ width: `${Math.min(100, (500 - (keystrokeMetrics.avgInterval || 0)) / 4)}%` }}></div>
                    </div>
                  </div>
                  
                  <div className="metric-box">
                    <div className="metric-header">
                      <Layers size={16} />
                      <span>Event Density</span>
                    </div>
                    <div className="metric-value">{sessionDuration > 0 ? Math.round(totalEvents / sessionDuration * 60) : 0}<span className="metric-unit">/min</span></div>
                    <div className="metric-bar">
                      <div className="metric-fill" style={{ width: `${Math.min(100, totalEvents / sessionDuration * 2)}%` }}></div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="demo-card bot-card">
                <div className="card-header">
                  <div className="card-title-group">
                    <div className="card-icon bot-icon">
                      <Bot size={20} />
                    </div>
                    <div>
                      <h3>Bot Simulator</h3>
                      <p>Test with automated patterns</p>
                    </div>
                  </div>
                  <button 
                    className="settings-toggle"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                  >
                    <Settings />
                  </button>
                </div>
                
                <div className="bot-type-grid">
                  {[
                    { id: 'basic', label: 'Basic', desc: 'Diagonal movement pattern' },
                    { id: 'linear', label: 'Linear', desc: 'Straight horizontal lines' },
                    { id: 'bezier', label: 'Grid', desc: 'Staircase/grid pattern' },
                    { id: 'teleport', label: 'Teleport', desc: 'Random position jumps' }
                  ].map(type => (
                    <button
                      key={type.id}
                      className={`bot-type-option ${botConfig.botType === type.id ? 'active' : ''}`}
                      onClick={() => setBotConfig(prev => ({ ...prev, botType: type.id }))}
                    >
                      <span className="bot-type-name">{type.label}</span>
                      <span className="bot-type-desc">{type.desc}</span>
                    </button>
                  ))}
                </div>
                
                {showAdvanced && (
                  <div className="bot-advanced">
                    <div className="speed-control">
                      <div className="speed-header">
                        <span>Simulation Speed</span>
                        <span className="speed-value">{botConfig.speed}%</span>
                      </div>
                      <input
                        type="range"
                        min="10"
                        max="100"
                        value={botConfig.speed}
                        onChange={(e) => setBotConfig(prev => ({ ...prev, speed: +e.target.value }))}
                        className="speed-slider"
                      />
                      <div className="speed-labels">
                        <span>Slow</span>
                        <span>Fast</span>
                      </div>
                    </div>
                  </div>
                )}
                
                <button 
                  className={`bot-control-btn ${isRunningBot ? 'running' : ''}`}
                  onClick={toggleBot}
                >
                  {isRunningBot ? (
                    <>
                      <Square size={18} />
                      <span>Stop Simulation</span>
                    </>
                  ) : (
                    <>
                      <Play size={18} />
                      <span>Start Bot Simulation</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Right Column - Analysis */}
            <div className="demo-column analysis-column">
              {/* Main Verdict */}
              <div className={`verdict-panel ${verdictClass}`}>
                <div className="verdict-ring-container">
                  <svg className="verdict-ring" viewBox="0 0 120 120">
                    <circle className="ring-track" cx="60" cy="60" r="54" />
                    <circle 
                      className="ring-progress"
                      cx="60" 
                      cy="60" 
                      r="54"
                      strokeDasharray={circumference}
                      strokeDashoffset={strokeDashoffset}
                    />
                  </svg>
                  <div className="verdict-center">
                    <span className="verdict-score">{Math.round(botScore)}</span>
                    <span className="verdict-unit">%</span>
                  </div>
                </div>
                
                <div className="verdict-info">
                  <div className={`verdict-label ${verdictClass}`}>
                    {verdictClass === 'human' && <User size={20} />}
                    {verdictClass === 'bot' && <Bot size={20} />}
                    {verdictClass === 'uncertain' && <AlertTriangle size={20} />}
                    {verdictClass === 'unknown' && <Eye size={20} />}
                    <span>{getVerdictText()}</span>
                  </div>
                  <p className="verdict-confidence">
                    {confidence > 0 ? `${confidence.toFixed(1)}% confidence` : 'Interact to start analysis'}
                  </p>
                </div>
                
                <div className="verdict-scale">
                  <div className="scale-bar">
                    <div className="scale-zone human">Human</div>
                    <div className="scale-zone uncertain">Uncertain</div>
                    <div className="scale-zone bot">Bot</div>
                  </div>
                  <div className="scale-marker" style={{ left: `${botScore}%` }}></div>
                </div>
              </div>

              {/* Feature Breakdown */}
              <div className="demo-card">
                <div className="card-header">
                  <div className="card-title-group">
                    <div className="card-icon">
                      <Activity size={20} />
                    </div>
                    <div>
                      <h3>Feature Analysis</h3>
                      <p>Component score breakdown</p>
                    </div>
                  </div>
                </div>
                
                <div className="feature-breakdown">
                  {[
                    { name: 'Mouse Velocity', key: 'velocity', Icon: Gauge },
                    { name: 'Path Analysis', key: 'path', Icon: Ruler },
                    { name: 'Timing Patterns', key: 'timing', Icon: Timer },
                    { name: 'Keystroke Rhythm', key: 'keystroke', Icon: Keyboard },
                    { name: 'Curvature Analysis', key: 'curvature', Icon: Crosshair }
                  ].map(feature => {
                    const value = componentScores[feature.key] ?? 0;
                    const percentage = Math.round(value * 100);
                    const status = percentage > 70 ? 'suspicious' : percentage > 40 ? 'uncertain' : 'normal';
                    return (
                      <div key={feature.key} className={`feature-item ${status}`}>
                        <div className="feature-header">
                          <span className="feature-icon-wrap"><feature.Icon size={14} /></span>
                          <span className="feature-name">{feature.name}</span>
                          <span className="feature-value">{percentage}%</span>
                        </div>
                        <div className="feature-bar-track">
                          <div 
                            className="feature-bar-fill"
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Triggers */}
              <div className="demo-card">
                <div className="card-header">
                  <div className="card-title-group">
                    <div className="card-icon">
                      <AlertTriangle size={20} />
                    </div>
                    <div>
                      <h3>Detection Signals</h3>
                      <p>Behavioral anomalies detected</p>
                    </div>
                  </div>
                </div>
                
                <div className="triggers-list">
                  {triggers.length > 0 ? (
                    triggers.slice(0, 5).map((trigger, i) => {
                      const triggerType = typeof trigger === 'object' ? trigger.type : 'info';
                      const triggerText = typeof trigger === 'object' ? trigger.text : trigger;
                      return (
                        <div key={i} className={`trigger-item ${triggerType}`}>
                          <TriggerIcon type={triggerType} />
                          <span>{triggerText}</span>
                        </div>
                      );
                    })
                  ) : (
                    <div className="triggers-empty">
                      <Shield size={32} />
                      <p>No anomalies detected</p>
                      <span>Behavioral patterns appear normal</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Score Timeline */}
              <div className="demo-card">
                <div className="card-header">
                  <div className="card-title-group">
                    <div className="card-icon">
                      <Activity size={20} />
                    </div>
                    <div>
                      <h3>Score Timeline</h3>
                      <p>Bot score over time</p>
                    </div>
                  </div>
                </div>
                
                <div className="timeline-container">
                  <div className="timeline-y-axis">
                    <span>100</span>
                    <span>50</span>
                    <span>0</span>
                  </div>
                  
                  <div className="timeline-chart">
                    {scoreHistory.length > 1 ? (
                      <svg viewBox="0 0 300 100" preserveAspectRatio="none">
                        <defs>
                          <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity="0.2" />
                            <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity="0" />
                          </linearGradient>
                        </defs>
                        
                        {/* Zone backgrounds */}
                        <rect x="0" y="0" width="300" height="30" fill="var(--bot-color)" opacity="0.06" />
                        <rect x="0" y="30" width="300" height="20" fill="var(--uncertain-color)" opacity="0.06" />
                        <rect x="0" y="50" width="300" height="50" fill="var(--human-color)" opacity="0.06" />
                        
                        {/* Threshold lines */}
                        <line x1="0" y1="30" x2="300" y2="30" stroke="var(--border-secondary)" strokeWidth="1" strokeDasharray="3,3" />
                        <line x1="0" y1="50" x2="300" y2="50" stroke="var(--border-secondary)" strokeWidth="1" strokeDasharray="3,3" />
                        
                        {/* Area fill */}
                        <polygon
                          fill="url(#chartGradient)"
                          points={`0,100 ${scoreHistory.map((p, i) => 
                            `${(i / (scoreHistory.length - 1)) * 300},${100 - p.score}`
                          ).join(' ')} 300,100`}
                        />
                        
                        {/* Line */}
                        <polyline
                          fill="none"
                          stroke="var(--accent-primary)"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          points={scoreHistory.map((p, i) => 
                            `${(i / (scoreHistory.length - 1)) * 300},${100 - p.score}`
                          ).join(' ')}
                        />
                        
                        {/* Current point */}
                        <circle
                          cx={300}
                          cy={100 - scoreHistory[scoreHistory.length - 1].score}
                          r="4"
                          fill="var(--accent-primary)"
                        />
                      </svg>
                    ) : (
                      <div className="timeline-empty">
                        <p>Collecting data points...</p>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="timeline-legend">
                  <div className="legend-item">
                    <span className="legend-dot bot"></span>
                    <span>Bot (70-100)</span>
                  </div>
                  <div className="legend-item">
                    <span className="legend-dot uncertain"></span>
                    <span>Uncertain (50-70)</span>
                  </div>
                  <div className="legend-item">
                    <span className="legend-dot human"></span>
                    <span>Human (0-50)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom CTA */}
          <div className="demo-cta-bar">
            <div className="demo-cta-content">
              <div className="demo-cta-text">
                <h3>Want to integrate this into your app?</h3>
                <p>Check out our SDK and API documentation for easy integration.</p>
              </div>
              <div className="demo-cta-actions">
                <Link to="/sdk" className="btn btn-primary">
                  <span>View SDK</span>
                  <ArrowRight size={18} />
                </Link>
                <Link to="/features" className="btn btn-secondary">
                  Learn More
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
