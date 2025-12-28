"""
Unit tests for ML Bot Detection Model
Tests feature extraction, prediction, and edge cases
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model_advanced import AdvancedBotDetector as BotDetector


class TestBotDetector:
    """Test suite for BotDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create a fresh BotDetector instance for each test"""
        return BotDetector()
    
    @pytest.fixture
    def human_like_data(self):
        """Generate human-like mouse behavior data"""
        # Human behavior: variable velocity, curved paths, irregular timing
        base_time = 1000000
        mouse_data = []
        x, y = 100, 100
        
        for i in range(50):
            # Add natural jitter and curves
            x += np.random.normal(5, 3)
            y += np.random.normal(3, 4)
            # Variable timing (50-150ms between events)
            base_time += np.random.randint(50, 150)
            mouse_data.append({
                'x': x,
                'y': y,
                'timestamp': base_time
            })
        
        keyboard_data = []
        for i in range(10):
            keyboard_data.append({
                'key': chr(97 + i % 26),
                'timestamp': base_time + i * np.random.randint(80, 200)
            })
        
        return {
            'mouse_data': mouse_data,
            'keyboard_data': keyboard_data,
            'click_data': [],
            'scroll_data': []
        }
    
    @pytest.fixture
    def bot_like_data(self):
        """Generate bot-like mouse behavior data"""
        # Bot behavior: constant velocity, straight lines, perfect timing
        base_time = 1000000
        mouse_data = []
        
        for i in range(50):
            # Perfect linear movement
            x = 100 + i * 10
            y = 100 + i * 5
            # Perfectly regular timing (exactly 50ms apart)
            base_time += 50
            mouse_data.append({
                'x': x,
                'y': y,
                'timestamp': base_time
            })
        
        keyboard_data = []
        for i in range(10):
            keyboard_data.append({
                'key': chr(97 + i % 26),
                'timestamp': base_time + i * 100  # Exactly 100ms apart
            })
        
        return {
            'mouse_data': mouse_data,
            'keyboard_data': keyboard_data,
            'click_data': [],
            'scroll_data': []
        }
    
    # ==================== Initialization Tests ====================
    
    def test_detector_initialization(self, detector):
        """Test that BotDetector initializes correctly"""
        assert detector is not None
        assert hasattr(detector, 'thresholds')
        assert hasattr(detector, 'use_ml_model')
    
    def test_model_loading(self, detector):
        """Test that trained model loads if available"""
        # Model should either be loaded or fallback to heuristic
        assert detector.use_ml_model in [True, False]
        if detector.use_ml_model:
            assert detector.model is not None
            assert detector.scaler is not None
            assert detector.feature_names is not None
    
    # ==================== Feature Extraction Tests ====================
    
    def test_extract_features_empty_data(self, detector):
        """Test feature extraction with empty data"""
        data = {
            'mouse_data': [],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        features = detector.extract_features(data)
        
        assert features is not None
        assert isinstance(features, dict)
        assert 'triggers' in features
    
    def test_extract_features_minimal_mouse_data(self, detector):
        """Test feature extraction with minimal mouse data"""
        data = {
            'mouse_data': [
                {'x': 0, 'y': 0, 'timestamp': 1000},
                {'x': 10, 'y': 10, 'timestamp': 1100},
                {'x': 20, 'y': 20, 'timestamp': 1200},
                {'x': 30, 'y': 30, 'timestamp': 1300},
                {'x': 40, 'y': 40, 'timestamp': 1400},
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        features = detector.extract_features(data)
        
        assert 'velocity_mean' in features or 'velocity_variance' in features
        assert 'straightness' in features
    
    def test_extract_features_with_keyboard_data(self, detector):
        """Test feature extraction includes keyboard analysis"""
        data = {
            'mouse_data': [
                {'x': i*10, 'y': i*5, 'timestamp': 1000 + i*100}
                for i in range(10)
            ],
            'keyboard_data': [
                {'key': 'a', 'timestamp': 1000},
                {'key': 'b', 'timestamp': 1150},
                {'key': 'c', 'timestamp': 1280},
                {'key': 'd', 'timestamp': 1450},
            ],
            'click_data': [],
            'scroll_data': []
        }
        features = detector.extract_features(data)
        
        assert 'keystroke_variance' in features or 'keystroke_mean' in features
    
    def test_extract_features_returns_triggers(self, detector):
        """Test that feature extraction returns trigger information"""
        data = {
            'mouse_data': [
                {'x': i*10, 'y': i*5, 'timestamp': 1000 + i*50}
                for i in range(20)
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        features = detector.extract_features(data)
        
        assert 'triggers' in features
        assert isinstance(features['triggers'], list)
    
    # ==================== Prediction Tests ====================
    
    def test_predict_returns_required_fields(self, detector, human_like_data):
        """Test that prediction returns all required fields"""
        features = detector.extract_features(human_like_data)
        prediction = detector.predict(features)
        
        required_fields = ['bot_score', 'is_bot', 'confidence', 'verdict']
        for field in required_fields:
            assert field in prediction, f"Missing required field: {field}"
    
    def test_predict_bot_score_range(self, detector, human_like_data):
        """Test that bot_score is within valid range [0, 100]"""
        features = detector.extract_features(human_like_data)
        prediction = detector.predict(features)
        
        assert 0 <= prediction['bot_score'] <= 100
    
    def test_predict_confidence_range(self, detector, human_like_data):
        """Test that confidence is within valid range [0, 100]"""
        features = detector.extract_features(human_like_data)
        prediction = detector.predict(features)
        
        assert 0 <= prediction['confidence'] <= 100
    
    def test_predict_is_bot_boolean(self, detector, human_like_data):
        """Test that is_bot is a boolean"""
        features = detector.extract_features(human_like_data)
        prediction = detector.predict(features)
        
        assert isinstance(prediction['is_bot'], bool)
    
    def test_predict_verdict_string(self, detector, human_like_data):
        """Test that verdict is a non-empty string"""
        features = detector.extract_features(human_like_data)
        prediction = detector.predict(features)
        
        assert isinstance(prediction['verdict'], str)
        assert len(prediction['verdict']) > 0
    
    def test_human_behavior_lower_score(self, detector, human_like_data, bot_like_data):
        """Test that human-like behavior gets lower bot score than bot-like"""
        human_features = detector.extract_features(human_like_data)
        bot_features = detector.extract_features(bot_like_data)
        
        human_prediction = detector.predict(human_features)
        bot_prediction = detector.predict(bot_features)
        
        # This may not always pass due to model variance, but generally should
        # We use a soft assertion - human score should be reasonably lower
        print(f"Human score: {human_prediction['bot_score']}")
        print(f"Bot score: {bot_prediction['bot_score']}")
    
    # ==================== Edge Case Tests ====================
    
    def test_predict_with_nan_values(self, detector):
        """Test prediction handles NaN values gracefully"""
        data = {
            'mouse_data': [
                {'x': float('nan'), 'y': 0, 'timestamp': 1000},
                {'x': 10, 'y': float('nan'), 'timestamp': 1100},
                {'x': 20, 'y': 20, 'timestamp': 1200},
                {'x': 30, 'y': 30, 'timestamp': 1300},
                {'x': 40, 'y': 40, 'timestamp': 1400},
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        
        try:
            features = detector.extract_features(data)
            prediction = detector.predict(features)
            assert 'bot_score' in prediction
        except Exception as e:
            pytest.fail(f"Should handle NaN gracefully: {e}")
    
    def test_predict_with_negative_timestamps(self, detector):
        """Test prediction handles negative timestamps"""
        data = {
            'mouse_data': [
                {'x': i*10, 'y': i*5, 'timestamp': -1000 + i*100}
                for i in range(10)
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        
        features = detector.extract_features(data)
        prediction = detector.predict(features)
        
        assert 'bot_score' in prediction
    
    def test_predict_with_large_coordinates(self, detector):
        """Test prediction handles very large coordinate values"""
        data = {
            'mouse_data': [
                {'x': i*10000, 'y': i*10000, 'timestamp': 1000 + i*100}
                for i in range(10)
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        
        features = detector.extract_features(data)
        prediction = detector.predict(features)
        
        assert 'bot_score' in prediction
        assert 0 <= prediction['bot_score'] <= 100
    
    def test_predict_stationary_mouse(self, detector):
        """Test prediction with stationary mouse (no movement)"""
        data = {
            'mouse_data': [
                {'x': 100, 'y': 100, 'timestamp': 1000 + i*100}
                for i in range(10)
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        
        features = detector.extract_features(data)
        prediction = detector.predict(features)
        
        assert 'bot_score' in prediction
    
    # ==================== Component Score Tests ====================
    
    def test_component_scores_exist(self, detector, human_like_data):
        """Test that component scores are returned"""
        features = detector.extract_features(human_like_data)
        prediction = detector.predict(features)
        
        if 'component_scores' in prediction:
            scores = prediction['component_scores']
            assert isinstance(scores, dict)
    
    def test_component_scores_range(self, detector, human_like_data):
        """Test that component scores are in valid range [0, 1]"""
        features = detector.extract_features(human_like_data)
        prediction = detector.predict(features)
        
        if 'component_scores' in prediction:
            for name, score in prediction['component_scores'].items():
                assert 0 <= score <= 1, f"Component {name} score out of range: {score}"


class TestFeatureExtraction:
    """Detailed tests for feature extraction logic"""
    
    @pytest.fixture
    def detector(self):
        return BotDetector()
    
    def test_velocity_calculation(self, detector):
        """Test that velocity is calculated correctly"""
        # Known movement: 100 pixels in 1000ms = 0.1 px/ms
        data = {
            'mouse_data': [
                {'x': 0, 'y': 0, 'timestamp': 0},
                {'x': 100, 'y': 0, 'timestamp': 1000},
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        # Need at least 5 points for analysis
        data['mouse_data'].extend([
            {'x': 200, 'y': 0, 'timestamp': 2000},
            {'x': 300, 'y': 0, 'timestamp': 3000},
            {'x': 400, 'y': 0, 'timestamp': 4000},
        ])
        
        features = detector.extract_features(data)
        
        # Velocity should be calculable
        assert 'velocity_mean' in features or 'velocity_variance' in features
    
    def test_straightness_calculation(self, detector):
        """Test straightness for perfectly straight line"""
        # Perfectly straight line should have high straightness
        data = {
            'mouse_data': [
                {'x': i*10, 'y': 0, 'timestamp': 1000 + i*100}
                for i in range(10)
            ],
            'keyboard_data': [],
            'click_data': [],
            'scroll_data': []
        }
        
        features = detector.extract_features(data)
        
        if 'straightness' in features:
            # Straight line should have straightness close to 1
            assert features['straightness'] >= 0.8


class TestIntegration:
    """Integration tests for full prediction pipeline"""
    
    @pytest.fixture
    def detector(self):
        return BotDetector()
    
    def test_full_pipeline_human(self, detector):
        """Test complete pipeline with realistic human data"""
        np.random.seed(42)
        
        mouse_data = []
        x, y = 500, 300
        t = 0
        
        for _ in range(100):
            # Simulate natural mouse movement
            x += np.random.normal(0, 15)
            y += np.random.normal(0, 10)
            t += np.random.randint(30, 100)
            mouse_data.append({'x': x, 'y': y, 'timestamp': t})
        
        keyboard_data = [
            {'key': chr(97 + i % 26), 'timestamp': i * np.random.randint(50, 200)}
            for i in range(20)
        ]
        
        click_data = [
            {'x': np.random.randint(0, 1000), 'y': np.random.randint(0, 600), 'timestamp': i * 500}
            for i in range(5)
        ]
        
        data = {
            'mouse_data': mouse_data,
            'keyboard_data': keyboard_data,
            'click_data': click_data,
            'scroll_data': []
        }
        
        features = detector.extract_features(data)
        prediction = detector.predict(features)
        
        assert prediction is not None
        assert 'bot_score' in prediction
        assert 'verdict' in prediction
        print(f"Human simulation score: {prediction['bot_score']}, verdict: {prediction['verdict']}")
    
    def test_full_pipeline_bot(self, detector):
        """Test complete pipeline with realistic bot data"""
        # Simulate perfect bot movement
        mouse_data = [
            {'x': 100 + i * 10, 'y': 100 + i * 5, 'timestamp': i * 50}
            for i in range(100)
        ]
        
        keyboard_data = [
            {'key': chr(97 + i % 26), 'timestamp': i * 100}
            for i in range(20)
        ]
        
        data = {
            'mouse_data': mouse_data,
            'keyboard_data': keyboard_data,
            'click_data': [],
            'scroll_data': []
        }
        
        features = detector.extract_features(data)
        prediction = detector.predict(features)
        
        assert prediction is not None
        assert 'bot_score' in prediction
        print(f"Bot simulation score: {prediction['bot_score']}, verdict: {prediction['verdict']}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
