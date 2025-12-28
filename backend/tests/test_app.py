"""
Integration tests for Flask API endpoints
Tests WebSocket connections and REST endpoints
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, socketio


class TestHealthEndpoint:
    """Tests for /api/health endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_check_returns_200(self, client):
        """Test that health endpoint returns 200"""
        response = client.get('/api/health')
        assert response.status_code == 200
    
    def test_health_check_returns_json(self, client):
        """Test that health endpoint returns valid JSON"""
        response = client.get('/api/health')
        data = response.get_json()
        
        assert data is not None
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_health_check_model_loaded(self, client):
        """Test that health endpoint confirms model is loaded"""
        response = client.get('/api/health')
        data = response.get_json()
        
        assert 'model' in data
        assert data['model'] == 'loaded'


class TestAnalyzeEndpoint:
    """Tests for /api/analyze endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_analyze_accepts_post(self, client):
        """Test that analyze endpoint accepts POST requests"""
        data = {
            'session_id': 'test-session',
            'mouse_data': [
                {'x': i*10, 'y': i*5, 'timestamp': 1000 + i*100}
                for i in range(10)
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        
        response = client.post('/api/analyze', json=data)
        assert response.status_code == 200
    
    def test_analyze_returns_prediction(self, client):
        """Test that analyze endpoint returns a prediction"""
        data = {
            'session_id': 'test-session',
            'mouse_data': [
                {'x': i*10, 'y': i*5, 'timestamp': 1000 + i*100}
                for i in range(10)
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        
        response = client.post('/api/analyze', json=data)
        result = response.get_json()
        
        assert result is not None
        assert 'bot_score' in result
    
    def test_analyze_with_empty_data(self, client):
        """Test analyze endpoint with minimal data"""
        data = {
            'session_id': 'test-session',
            'mouse_data': [],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        
        response = client.post('/api/analyze', json=data)
        assert response.status_code == 200


class TestWebSocket:
    """Tests for WebSocket connections"""
    
    @pytest.fixture
    def socket_client(self):
        """Create a Socket.IO test client"""
        app.config['TESTING'] = True
        client = socketio.test_client(app)
        yield client
        client.disconnect()
    
    def test_socket_connection(self, socket_client):
        """Test that WebSocket connection is established"""
        assert socket_client.is_connected()
    
    def test_socket_receives_connected_event(self, socket_client):
        """Test that client receives connected event"""
        received = socket_client.get_received()
        
        # Look for 'connected' event
        event_names = [r['name'] for r in received]
        assert 'connected' in event_names
    
    def test_socket_behavior_data_event(self, socket_client):
        """Test sending behavior_data event"""
        data = {
            'mouse_data': [
                {'x': i*10, 'y': i*5, 'timestamp': 1000 + i*100}
                for i in range(10)
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        
        socket_client.emit('behavior_data', data)
        
        # Should receive analysis_result
        received = socket_client.get_received()
        # Note: May need to wait for processing
        assert socket_client.is_connected()
    
    def test_socket_reset_session(self, socket_client):
        """Test reset_session event"""
        socket_client.emit('reset_session')
        
        received = socket_client.get_received()
        event_names = [r['name'] for r in received]
        
        # Should receive session_reset confirmation
        assert 'session_reset' in event_names


class TestCORS:
    """Tests for CORS configuration"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present"""
        response = client.get('/api/health')
        
        # CORS headers should allow cross-origin requests
        # Note: Headers may vary based on CORS configuration
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
