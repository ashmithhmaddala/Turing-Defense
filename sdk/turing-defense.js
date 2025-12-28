/**
 * Turing Defense - Bot Detection SDK
 * Embed this script on any website to detect bots using behavioral biometrics.
 * 
 * Usage:
 *   <script src="https://your-domain.com/sdk/turing-defense.js" 
 *           data-site-key="your-site-key"
 *           data-endpoint="https://your-api.com"></script>
 * 
 * API:
 *   TuringDefense.getScore()     - Get current bot score (0-100)
 *   TuringDefense.isBot()        - Returns true if likely a bot
 *   TuringDefense.onResult(fn)   - Callback when analysis is ready
 *   TuringDefense.reset()        - Reset session data
 *   TuringDefense.getToken()     - Get verification token for form submission
 * 
 * Built by Ashmith Maddala
 * https://github.com/ashmithmaddala
 */

(function(window, document) {
  'use strict';

  // Configuration
  const CONFIG = {
    endpoint: null,
    siteKey: null,
    sendInterval: 1000,      // How often to send data (ms)
    minDataPoints: 20,       // Minimum points before first analysis
    debug: false,
    autoStart: true
  };

  // State
  const state = {
    initialized: false,
    sessionId: null,
    socket: null,
    
    // Collected data
    mouseData: [],
    keystrokeData: [],
    clickData: [],
    scrollData: [],
    
    // Timing
    lastKeyTime: null,
    sessionStart: Date.now(),
    
    // Results
    latestResult: null,
    callbacks: [],
    
    // Intervals
    sendIntervalId: null
  };

  // Generate unique session ID
  function generateSessionId() {
    return 'td_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
  }

  // Initialize from script tag attributes
  function initFromScriptTag() {
    const scripts = document.getElementsByTagName('script');
    for (let i = scripts.length - 1; i >= 0; i--) {
      const script = scripts[i];
      if (script.src && script.src.includes('turing-defense')) {
        CONFIG.endpoint = script.getAttribute('data-endpoint') || CONFIG.endpoint;
        CONFIG.siteKey = script.getAttribute('data-site-key') || CONFIG.siteKey;
        CONFIG.debug = script.getAttribute('data-debug') === 'true';
        CONFIG.autoStart = script.getAttribute('data-auto-start') !== 'false';
        break;
      }
    }
  }

  // Logging
  function log(...args) {
    if (CONFIG.debug) {
      console.log('[TuringDefense]', ...args);
    }
  }

  // Mouse move handler
  function handleMouseMove(e) {
    state.mouseData.push({
      x: e.clientX,
      y: e.clientY,
      timestamp: Date.now(),
      target: e.target.tagName
    });
    
    // Keep last 500 points to avoid memory bloat
    if (state.mouseData.length > 500) {
      state.mouseData = state.mouseData.slice(-500);
    }
  }

  // Mouse click handler
  function handleClick(e) {
    state.clickData.push({
      x: e.clientX,
      y: e.clientY,
      timestamp: Date.now(),
      button: e.button,
      target: e.target.tagName,
      targetId: e.target.id || null,
      targetClass: e.target.className || null
    });
  }

  // Keydown handler
  function handleKeyDown(e) {
    const now = Date.now();
    
    // Don't capture actual key values for privacy - just timing and metadata
    state.keystrokeData.push({
      timestamp: now,
      type: 'keydown',
      keyCode: e.keyCode,
      isSpecial: e.ctrlKey || e.altKey || e.metaKey,
      interval: state.lastKeyTime ? now - state.lastKeyTime : null,
      target: e.target.tagName
    });
    
    state.lastKeyTime = now;
    
    if (state.keystrokeData.length > 500) {
      state.keystrokeData = state.keystrokeData.slice(-500);
    }
  }

  // Keyup handler
  function handleKeyUp(e) {
    state.keystrokeData.push({
      timestamp: Date.now(),
      type: 'keyup',
      keyCode: e.keyCode,
      target: e.target.tagName
    });
  }

  // Scroll handler
  function handleScroll(e) {
    state.scrollData.push({
      timestamp: Date.now(),
      deltaY: e.deltaY || window.scrollY,
      deltaX: e.deltaX || window.scrollX,
      type: e.type
    });
    
    if (state.scrollData.length > 200) {
      state.scrollData = state.scrollData.slice(-200);
    }
  }

  // Attach event listeners
  function attachListeners() {
    document.addEventListener('mousemove', handleMouseMove, { passive: true });
    document.addEventListener('click', handleClick, { passive: true });
    document.addEventListener('keydown', handleKeyDown, { passive: true });
    document.addEventListener('keyup', handleKeyUp, { passive: true });
    document.addEventListener('wheel', handleScroll, { passive: true });
    document.addEventListener('scroll', handleScroll, { passive: true });
    
    log('Event listeners attached');
  }

  // Detach event listeners
  function detachListeners() {
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('click', handleClick);
    document.removeEventListener('keydown', handleKeyDown);
    document.removeEventListener('keyup', handleKeyUp);
    document.removeEventListener('wheel', handleScroll);
    document.removeEventListener('scroll', handleScroll);
  }

  // Send data to backend
  async function sendData() {
    const totalPoints = state.mouseData.length + state.keystrokeData.length + 
                        state.clickData.length + state.scrollData.length;
    
    if (totalPoints < CONFIG.minDataPoints) {
      log('Not enough data points yet:', totalPoints);
      return;
    }

    const payload = {
      session_id: state.sessionId,
      site_key: CONFIG.siteKey,
      timestamp: Date.now(),
      session_duration: Date.now() - state.sessionStart,
      data: {
        mouse: [...state.mouseData],
        keystrokes: [...state.keystrokeData],
        clicks: [...state.clickData],
        scroll: [...state.scrollData]
      },
      metadata: {
        url: window.location.href,
        referrer: document.referrer,
        userAgent: navigator.userAgent,
        screenWidth: window.screen.width,
        screenHeight: window.screen.height,
        language: navigator.language,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
      }
    };

    // Clear sent data
    state.mouseData = [];
    state.keystrokeData = [];
    state.clickData = [];
    state.scrollData = [];

    try {
      const response = await fetch(`${CONFIG.endpoint}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Site-Key': CONFIG.siteKey || ''
        },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        const result = await response.json();
        state.latestResult = result;
        
        log('Analysis result:', result);
        
        // Trigger callbacks
        state.callbacks.forEach(cb => {
          try {
            cb(result);
          } catch (e) {
            console.error('[TuringDefense] Callback error:', e);
          }
        });
      }
    } catch (error) {
      log('Error sending data:', error);
    }
  }

  // WebSocket connection (optional, for real-time)
  function connectWebSocket() {
    if (!CONFIG.endpoint) return;
    
    const wsUrl = CONFIG.endpoint.replace('http', 'ws') + '/ws';
    
    try {
      state.socket = new WebSocket(wsUrl);
      
      state.socket.onopen = () => {
        log('WebSocket connected');
        state.socket.send(JSON.stringify({
          type: 'init',
          session_id: state.sessionId,
          site_key: CONFIG.siteKey
        }));
      };
      
      state.socket.onmessage = (event) => {
        try {
          const result = JSON.parse(event.data);
          state.latestResult = result;
          state.callbacks.forEach(cb => cb(result));
        } catch (e) {}
      };
      
      state.socket.onclose = () => {
        log('WebSocket closed, falling back to HTTP');
      };
    } catch (e) {
      log('WebSocket not available, using HTTP');
    }
  }

  // Public API
  const TuringDefense = {
    /**
     * Initialize the SDK manually (if auto-start is disabled)
     */
    init: function(options = {}) {
      if (state.initialized) {
        log('Already initialized');
        return this;
      }

      // Merge options
      Object.assign(CONFIG, options);
      
      // Generate session ID
      state.sessionId = generateSessionId();
      state.sessionStart = Date.now();
      
      // Attach listeners
      attachListeners();
      
      // Start sending data periodically
      state.sendIntervalId = setInterval(sendData, CONFIG.sendInterval);
      
      // Try WebSocket for real-time (optional)
      if (CONFIG.endpoint) {
        connectWebSocket();
      }
      
      state.initialized = true;
      log('Initialized with session:', state.sessionId);
      
      return this;
    },

    /**
     * Get current bot score (0-100, higher = more likely bot)
     */
    getScore: function() {
      if (!state.latestResult) return null;
      return state.latestResult.bot_score || state.latestResult.prediction?.bot_score || 0;
    },

    /**
     * Check if current user is likely a bot
     */
    isBot: function() {
      if (!state.latestResult) return null;
      const score = this.getScore();
      return score !== null ? score > 50 : null;
    },

    /**
     * Get confidence level (0-100)
     */
    getConfidence: function() {
      if (!state.latestResult) return null;
      return state.latestResult.confidence || state.latestResult.prediction?.confidence || 0;
    },

    /**
     * Get full analysis result
     */
    getResult: function() {
      return state.latestResult;
    },

    /**
     * Register callback for when analysis results arrive
     */
    onResult: function(callback) {
      if (typeof callback === 'function') {
        state.callbacks.push(callback);
      }
      return this;
    },

    /**
     * Get verification token for form submission
     * Include this token in your form to verify on the server
     */
    getToken: function() {
      if (!state.latestResult) return null;
      return state.latestResult.token || null;
    },

    /**
     * Reset session and collected data
     */
    reset: function() {
      state.mouseData = [];
      state.keystrokeData = [];
      state.clickData = [];
      state.scrollData = [];
      state.latestResult = null;
      state.lastKeyTime = null;
      state.sessionId = generateSessionId();
      state.sessionStart = Date.now();
      
      log('Session reset');
      return this;
    },

    /**
     * Stop collecting data and cleanup
     */
    destroy: function() {
      detachListeners();
      
      if (state.sendIntervalId) {
        clearInterval(state.sendIntervalId);
      }
      
      if (state.socket) {
        state.socket.close();
      }
      
      state.initialized = false;
      log('Destroyed');
    },

    /**
     * Get session ID
     */
    getSessionId: function() {
      return state.sessionId;
    },

    /**
     * Get data collection stats
     */
    getStats: function() {
      return {
        sessionId: state.sessionId,
        sessionDuration: Date.now() - state.sessionStart,
        mousePoints: state.mouseData.length,
        keystrokes: state.keystrokeData.length,
        clicks: state.clickData.length,
        scrollEvents: state.scrollData.length,
        hasResult: state.latestResult !== null
      };
    },

    /**
     * Force send current data for analysis
     */
    analyze: function() {
      return sendData();
    },

    // Expose version
    version: '1.0.0'
  };

  // Auto-initialize on DOM ready
  function autoInit() {
    initFromScriptTag();
    
    if (CONFIG.autoStart && CONFIG.endpoint) {
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => TuringDefense.init());
      } else {
        TuringDefense.init();
      }
    }
  }

  // Expose to global scope
  window.TuringDefense = TuringDefense;

  // Auto-init
  autoInit();

})(window, document);
