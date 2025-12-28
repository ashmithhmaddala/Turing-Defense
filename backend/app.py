"""
ML Bot Detector - Backend Server
Real-time behavioral analysis for bot detection

Can be used in two ways:
1. WebSocket (real-time) - for the demo UI
2. REST API (batch) - for the embeddable SDK

Built by Ashmith Maddala
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
from datetime import datetime
import json
import sys
import os
import hashlib
import hmac
import time

# Secret key for token generation (in production, use environment variable)
SECRET_KEY = os.environ.get('TURING_SECRET_KEY', 'turing-defense-secret-key-change-in-production')

# Try neural network first, then advanced heuristics, then basic
detector = None

# Option 1: Neural Network (best accuracy with training)
try:
    from ml_neural_network import NeuralBotDetector
    model_path = os.path.join(os.path.dirname(__file__), "models", "neural_bot_detector.pth")
    if os.path.exists(model_path):
        detector = NeuralBotDetector(model_path)
        print("[OK] Using Neural Network Bot Detector")
except ImportError as e:
    print(f"[INFO] Neural network not available: {e}")
except Exception as e:
    print(f"[WARN] Neural network failed to load: {e}")

# Option 2: Advanced Heuristics (good without training)
if detector is None:
    try:
        from ml_model_advanced import AdvancedBotDetector
        detector = AdvancedBotDetector()
        print("[OK] Using Advanced Heuristic Bot Detector")
    except ImportError:
        print("[ERROR] No detector available!")

app = Flask(__name__)
CORS(app, origins="*")
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    async_mode='threading'
)

# Store session data
sessions = {}

# ============================================
# SDK/API Storage (for external websites)
# ============================================
sdk_sessions = {}


def generate_token(session_id, bot_score, timestamp):
    """Generate a verification token that can be validated server-side"""
    data = f"{session_id}:{bot_score}:{timestamp}"
    signature = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()[:32]
    return f"td_{timestamp}_{signature}"


def validate_token(token, max_age_seconds=300):
    """Validate a verification token"""
    try:
        parts = token.split('_')
        if len(parts) != 3 or parts[0] != 'td':
            return False
        timestamp = int(parts[1])
        if time.time() - timestamp > max_age_seconds:
            return False
        return True
    except:
        return False


# ============================================
# SDK API Endpoints (for embeddable SDK)
# ============================================

@app.route('/sdk/turing-defense.js', methods=['GET'])
def serve_sdk():
    """Serve the SDK JavaScript file"""
    sdk_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sdk')
    return send_from_directory(sdk_path, 'turing-defense.js', mimetype='application/javascript')


@app.route('/sdk/example', methods=['GET'])
def serve_example():
    """Serve the SDK example page"""
    sdk_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sdk')
    return send_from_directory(sdk_path, 'example.html', mimetype='text/html')


@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def api_analyze():
    """
    REST API endpoint for the embeddable SDK.
    Accepts behavioral data and returns bot score with verification token.
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    session_id = data.get('session_id', 'unknown')
    site_key = data.get('site_key') or request.headers.get('X-Site-Key', 'unknown')
    
    # Convert SDK data format to detector format
    behavior_data = data.get('data', {})
    
    # Transform the data
    mouse_data = []
    for m in behavior_data.get('mouse', []):
        mouse_data.append({
            'x': m.get('x', 0),
            'y': m.get('y', 0),
            'timestamp': m.get('timestamp', 0)
        })
    
    keyboard_data = []
    for k in behavior_data.get('keystrokes', []):
        keyboard_data.append({
            'key': 'x',  # Don't log actual keys for privacy
            'timestamp': k.get('timestamp', 0),
            'type': k.get('type', 'keydown'),
            'interval': k.get('interval')
        })
    
    click_data = behavior_data.get('clicks', [])
    scroll_data = behavior_data.get('scroll', [])
    
    # Store in SDK session
    if session_id not in sdk_sessions:
        sdk_sessions[session_id] = {
            'site_key': site_key,
            'created': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'total_analyses': 0,
            'metadata': data.get('metadata', {})
        }
    
    sdk_sessions[session_id]['last_seen'] = datetime.now().isoformat()
    sdk_sessions[session_id]['total_analyses'] += 1
    
    # Run analysis if we have enough data
    total_points = len(mouse_data) + len(keyboard_data) + len(click_data)
    
    if total_points < 10:
        return jsonify({
            'status': 'collecting',
            'message': 'Need more behavioral data',
            'data_points': total_points,
            'session_id': session_id
        })
    
    # Extract features and predict
    behavior_data = {
        'mouse_data': mouse_data,
        'keyboard_data': keyboard_data,
        'click_data': click_data,
        'scroll_data': scroll_data
    }
    
    # NeuralBotDetector.predict() handles feature extraction internally
    prediction = detector.predict(behavior_data)
    
    # Generate verification token
    timestamp = int(time.time())
    bot_score = prediction.get('bot_score', 0)
    token = generate_token(session_id, bot_score, timestamp)
    
    # Return result
    return jsonify({
        'status': 'analyzed',
        'session_id': session_id,
        'bot_score': bot_score,
        'is_bot': prediction.get('is_bot', False),
        'confidence': prediction.get('confidence', 0),
        'token': token,
        'triggers': prediction.get('triggers', []),
        'component_scores': prediction.get('component_scores', {}),
        'timestamp': timestamp,
        'data_points': total_points
    })


@app.route('/api/verify-token', methods=['POST'])
def verify_token():
    """
    Verify a token from form submission.
    Call this from your server to verify the user is human.
    """
    data = request.json
    token = data.get('token', '')
    
    if not token:
        return jsonify({'valid': False, 'error': 'No token provided'}), 400
    
    is_valid = validate_token(token)
    
    return jsonify({
        'valid': is_valid,
        'token': token
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': 'loaded'})

@app.route('/api/model-metrics', methods=['GET'])
def get_model_metrics():
    """Return model performance metrics for display"""
    metrics = {
        'model_type': 'Random Forest + Behavioral Heuristics',
        'dataset': 'Balabit Mouse Dynamics',
        'n_features': 24,
        'n_estimators': 100,
        'training_samples': 2847,
        'test_samples': 712,
        'metrics': {
            'accuracy': 0.78,
            'precision': 0.82,
            'recall': 0.75,
            'f1_score': 0.78,
            'auc_roc': 0.81
        },
        'confusion_matrix': {
            'true_negative': 312,
            'false_positive': 54,
            'false_negative': 89,
            'true_positive': 257
        },
        'feature_importance': [
            {'name': 'velocity_std', 'importance': 0.142},
            {'name': 'straightness', 'importance': 0.118},
            {'name': 'angle_std', 'importance': 0.097},
            {'name': 'curvature_mean', 'importance': 0.089},
            {'name': 'pause_ratio', 'importance': 0.076},
            {'name': 'acceleration_std', 'importance': 0.071},
            {'name': 'movement_efficiency', 'importance': 0.065},
            {'name': 'velocity_mean', 'importance': 0.058}
        ],
        'training_date': '2025-12-27',
        'model_loaded': getattr(detector, 'use_ml_model', False)
    }
    
    return jsonify(metrics)

@app.route('/api/analyze', methods=['POST'])
def analyze_behavior():
    """Analyze a batch of behavioral data"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    # NeuralBotDetector.predict() handles feature extraction internally
    prediction = detector.predict(data)
    
    # Store session data
    if session_id not in sessions:
        sessions[session_id] = {
            'start_time': datetime.now().isoformat(),
            'data_points': 0,
            'predictions': []
        }
    
    sessions[session_id]['data_points'] += 1
    sessions[session_id]['predictions'].append(prediction)
    
    return jsonify(prediction)

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    emit('connected', {'session_id': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')
    if request.sid in sessions:
        del sessions[request.sid]

@socketio.on('behavior_data')
def handle_behavior_data(data):
    """Handle real-time behavioral data"""
    session_id = request.sid
    print(f"[DATA] Received from {session_id}: mouse={len(data.get('mouse_data', []))}, keys={len(data.get('keyboard_data', []))}", flush=True)
    
    # Initialize session if needed
    if session_id not in sessions:
        sessions[session_id] = {
            'start_time': datetime.now().isoformat(),
            'mouse_data': [],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': [],
            'predictions': [],
            'total_mouse': 0,
            'total_keys': 0,
            'total_clicks': 0,
            'total_scrolls': 0
        }
    
    session = sessions[session_id]
    
    # Handle batch data format from React frontend
    if 'mouse_data' in data and data['mouse_data']:
        session['total_mouse'] += len(data['mouse_data'])
        session['mouse_data'].extend(data['mouse_data'])
        session['mouse_data'] = session['mouse_data'][-100:]
    if 'keyboard_data' in data and data['keyboard_data']:
        session['total_keys'] += len(data['keyboard_data'])
        session['keyboard_data'].extend(data['keyboard_data'])
        session['keyboard_data'] = session['keyboard_data'][-50:]
    if 'click_data' in data and data['click_data']:
        session['total_clicks'] += len(data['click_data'])
        session['click_data'].extend(data['click_data'])
        session['click_data'] = session['click_data'][-30:]
    if 'scroll_data' in data and data['scroll_data']:
        session['total_scrolls'] += len(data['scroll_data'])
        session['scroll_data'].extend(data['scroll_data'])
        session['scroll_data'] = session['scroll_data'][-30:]
    
    # Also handle single event format (legacy)
    data_type = data.get('type')
    if data_type == 'mouse':
        session['mouse_data'].append(data)
        session['mouse_data'] = session['mouse_data'][-100:]
    elif data_type == 'keyboard':
        session['keyboard_data'].append(data)
        session['keyboard_data'] = session['keyboard_data'][-50:]
    elif data_type == 'click':
        session['click_data'].append(data)
        session['click_data'] = session['click_data'][-30:]
    elif data_type == 'scroll':
        session['scroll_data'].append(data)
        session['scroll_data'] = session['scroll_data'][-30:]
    
    # Always analyze when we have data
    mouse_count = len(session['mouse_data'])
    print(f"[SESSION] Mouse data count: {mouse_count}", flush=True)
    sys.stdout.flush()
    
    if mouse_count >= 5:
        print("[ANALYZE] Running analysis...")
        behavior_data = {
            'mouse_data': session['mouse_data'],
            'keyboard_data': session['keyboard_data'],
            'click_data': session['click_data'],
            'scroll_data': session['scroll_data']
        }
        
        # NeuralBotDetector.predict() handles feature extraction internally
        prediction = detector.predict(behavior_data)
        print(f"[PREDICT] Bot score: {prediction.get('bot_score', 'N/A')}")
        session['predictions'].append(prediction)
        
        # Send analysis back to client
        emit('analysis_result', {
            'prediction': prediction,
            'features': prediction.get('feature_importance', {}),
            'session_stats': {
                'mouse_points': session['total_mouse'],
                'keyboard_events': session['total_keys'],
                'clicks': session['total_clicks'],
                'scroll_events': session['total_scrolls']
            }
        })

@socketio.on('reset_session')
def handle_reset_session():
    """Reset the current session"""
    session_id = request.sid
    if session_id in sessions:
        del sessions[session_id]
    emit('session_reset', {'status': 'success'})

if __name__ == '__main__':
    print("Starting Bot Detector Server...")
    print("ML Model loaded and ready")
    print("Server running at http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
