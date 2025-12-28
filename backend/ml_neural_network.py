"""
Neural Network Bot Detection Model
Production-grade deep learning for human/bot classification
"""

import numpy as np
import json
import os
from collections import Counter

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. Run: pip install torch")


# ============================================================================
# RECOMMENDED DATASETS FOR BOT DETECTION
# ============================================================================
"""
PUBLIC DATASETS FOR TRAINING:

1. **Balabit Mouse Dynamics Dataset** (BEST FOR THIS PROJECT)
   - URL: https://github.com/balabit/Mouse-Dynamics-Challenge
   - Contains: Mouse movements from 10 users, ~1.5M samples
   - Features: x, y, timestamp, button state
   - License: Open for research
   - Use case: Learn genuine human mouse patterns

2. **CMU Keystroke Dynamics Dataset**
   - URL: https://www.cs.cmu.edu/~keystroke/
   - Contains: Typing patterns from 51 subjects
   - Features: Key press/release times, digraph latencies
   - Use case: Keyboard-based authentication

3. **Kaggle Bot Detection Datasets**
   - "Bot or Not" challenge data
   - Various click/scroll behavior datasets
   - Search: kaggle.com/datasets?search=bot+detection

4. **CRAWDAD Wireless Dataset** (for timing patterns)
   - URL: crawdad.org
   - Contains: Various behavioral timing data

5. **Your Own Data Collection** (RECOMMENDED)
   - Use the DataCollector class below
   - Collect from real users with consent
   - Label manually or use CAPTCHA as ground truth

HOW TO USE:
1. Download Balabit dataset: git clone https://github.com/balabit/Mouse-Dynamics-Challenge
2. Place in backend/datasets/balabit/
3. Run: python ml_neural_network.py --train --dataset balabit
"""


# ============================================================================
# EXPANDED FEATURE EXTRACTION (40+ features)
# ============================================================================

class AdvancedFeatureExtractor:
    """
    Extracts 40+ behavioral features for neural network input
    """
    
    def __init__(self):
        self.feature_names = []
        self._build_feature_list()
    
    def _build_feature_list(self):
        """Define all features for documentation and validation"""
        self.feature_names = [
            # Mouse Kinematics (12 features)
            'velocity_mean', 'velocity_std', 'velocity_max', 'velocity_cv',
            'acceleration_mean', 'acceleration_std', 'acceleration_max', 'acceleration_cv',
            'jerk_mean', 'jerk_std', 'jerk_max', 'jerk_cv',
            
            # Mouse Geometry (10 features)
            'path_straightness', 'path_efficiency', 'curvature_mean', 'curvature_std',
            'curvature_entropy', 'direction_changes', 'angular_velocity_mean', 'angular_velocity_std',
            'bezier_fit_error', 'convex_hull_ratio',
            
            # Mouse Temporal (8 features)
            'timing_entropy', 'pause_count', 'pause_duration_mean', 'pause_duration_std',
            'movement_duration', 'idle_ratio', 'burst_count', 'burst_intensity',
            
            # Mouse Micro-patterns (6 features)
            'micro_corrections', 'direction_reversals', 'hesitation_count',
            'overshoot_count', 'tremor_score', 'smoothness_score',
            
            # Keyboard Features (8 features)
            'keystroke_interval_mean', 'keystroke_interval_std', 'keystroke_cv',
            'digraph_latency_mean', 'digraph_latency_std', 'typing_speed',
            'key_hold_mean', 'key_hold_std',
            
            # Click Features (4 features)
            'click_interval_mean', 'click_interval_std', 'double_click_ratio',
            'click_precision',
            
            # Scroll Features (4 features)
            'scroll_velocity_mean', 'scroll_velocity_std', 'scroll_direction_changes',
            'scroll_burst_ratio',
            
            # Session Features (4 features)
            'session_duration', 'activity_density', 'input_diversity',
            'behavioral_consistency'
        ]
    
    def extract(self, data):
        """Extract all features from behavioral data"""
        features = np.zeros(len(self.feature_names))
        
        mouse_data = data.get('mouse_data', [])
        keyboard_data = data.get('keyboard_data', [])
        click_data = data.get('click_data', [])
        scroll_data = data.get('scroll_data', [])
        
        idx = 0
        
        # Mouse Kinematics
        kin_features = self._extract_kinematics(mouse_data)
        features[idx:idx+12] = kin_features
        idx += 12
        
        # Mouse Geometry
        geo_features = self._extract_geometry(mouse_data)
        features[idx:idx+10] = geo_features
        idx += 10
        
        # Mouse Temporal
        temp_features = self._extract_temporal(mouse_data)
        features[idx:idx+8] = temp_features
        idx += 8
        
        # Mouse Micro-patterns
        micro_features = self._extract_micropatterns(mouse_data)
        features[idx:idx+6] = micro_features
        idx += 6
        
        # Keyboard Features
        key_features = self._extract_keyboard(keyboard_data)
        features[idx:idx+8] = key_features
        idx += 8
        
        # Click Features
        click_features = self._extract_clicks(click_data)
        features[idx:idx+4] = click_features
        idx += 4
        
        # Scroll Features
        scroll_features = self._extract_scroll(scroll_data)
        features[idx:idx+4] = scroll_features
        idx += 4
        
        # Session Features
        session_features = self._extract_session(data)
        features[idx:idx+4] = session_features
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        return features
    
    def _extract_kinematics(self, mouse_data):
        """Extract velocity, acceleration, jerk features"""
        features = np.zeros(12)
        
        if len(mouse_data) < 5:
            return features
        
        # Calculate velocities
        positions = np.array([[p['x'], p['y']] for p in mouse_data])
        times = np.array([p['timestamp'] for p in mouse_data])
        
        dt = np.diff(times) / 1000.0  # Convert to seconds
        dt = np.maximum(dt, 0.001)  # Prevent division by zero
        
        dx = np.diff(positions[:, 0])
        dy = np.diff(positions[:, 1])
        
        velocities = np.sqrt(dx**2 + dy**2) / dt
        accelerations = np.diff(velocities) / dt[1:]
        jerks = np.diff(accelerations) / dt[2:] if len(dt) > 2 else np.array([0])
        
        # Velocity stats
        features[0] = np.mean(velocities)
        features[1] = np.std(velocities)
        features[2] = np.max(velocities) if len(velocities) > 0 else 0
        features[3] = np.std(velocities) / (np.mean(velocities) + 1e-6)  # CV
        
        # Acceleration stats
        if len(accelerations) > 0:
            features[4] = np.mean(np.abs(accelerations))
            features[5] = np.std(accelerations)
            features[6] = np.max(np.abs(accelerations))
            features[7] = np.std(accelerations) / (np.mean(np.abs(accelerations)) + 1e-6)
        
        # Jerk stats
        if len(jerks) > 0:
            features[8] = np.mean(np.abs(jerks))
            features[9] = np.std(jerks)
            features[10] = np.max(np.abs(jerks))
            features[11] = np.std(jerks) / (np.mean(np.abs(jerks)) + 1e-6)
        
        return features
    
    def _extract_geometry(self, mouse_data):
        """Extract geometric path features"""
        features = np.zeros(10)
        
        if len(mouse_data) < 3:
            return features
        
        positions = np.array([[p['x'], p['y']] for p in mouse_data])
        
        # Path straightness
        total_dist = sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
        direct_dist = np.sqrt(np.sum((positions[-1] - positions[0])**2))
        features[0] = direct_dist / (total_dist + 1e-6)  # straightness
        features[1] = direct_dist / (total_dist + 1e-6)  # efficiency (same for now)
        
        # Curvature
        if len(positions) >= 3:
            dx = np.gradient(positions[:, 0])
            dy = np.gradient(positions[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
            curvature = curvature[~np.isnan(curvature) & ~np.isinf(curvature)]
            
            if len(curvature) > 0:
                features[2] = np.mean(curvature)
                features[3] = np.std(curvature)
                
                # Curvature entropy
                hist, _ = np.histogram(curvature, bins=10, density=True)
                hist = hist[hist > 0]
                features[4] = -np.sum(hist * np.log(hist + 1e-10))
        
        # Direction changes
        dx = np.diff(positions[:, 0])
        dy = np.diff(positions[:, 1])
        angles = np.arctan2(dy, dx)
        angle_changes = np.abs(np.diff(angles))
        angle_changes = np.where(angle_changes > np.pi, 2*np.pi - angle_changes, angle_changes)
        features[5] = np.sum(angle_changes > 0.5)  # Count significant direction changes
        
        # Angular velocity
        if len(angle_changes) > 0:
            features[6] = np.mean(angle_changes)
            features[7] = np.std(angle_changes)
        
        # Bezier fit error (simplified)
        features[8] = self._bezier_fit_error(positions)
        
        # Convex hull ratio
        try:
            from scipy.spatial import ConvexHull
            if len(positions) >= 3:
                hull = ConvexHull(positions)
                features[9] = hull.area / (total_dist**2 + 1e-6)
        except:
            features[9] = 0.5
        
        return features
    
    def _bezier_fit_error(self, positions):
        """Calculate how well the path fits a Bezier curve"""
        if len(positions) < 4:
            return 0.0
        
        # Fit cubic Bezier
        t = np.linspace(0, 1, len(positions))
        
        # Control points (simplified: use start, 1/3, 2/3, end)
        p0 = positions[0]
        p3 = positions[-1]
        p1 = positions[len(positions)//3]
        p2 = positions[2*len(positions)//3]
        
        # Bezier curve points
        bezier = np.array([
            (1-ti)**3 * p0 + 3*(1-ti)**2*ti * p1 + 3*(1-ti)*ti**2 * p2 + ti**3 * p3
            for ti in t
        ])
        
        # Calculate error
        error = np.mean(np.sqrt(np.sum((positions - bezier)**2, axis=1)))
        return error
    
    def _extract_temporal(self, mouse_data):
        """Extract timing-related features"""
        features = np.zeros(8)
        
        if len(mouse_data) < 3:
            return features
        
        times = np.array([p['timestamp'] for p in mouse_data])
        intervals = np.diff(times)
        
        # Timing entropy
        if len(intervals) > 0:
            hist, _ = np.histogram(intervals, bins=10, density=True)
            hist = hist[hist > 0]
            features[0] = -np.sum(hist * np.log(hist + 1e-10))
        
        # Pause detection (interval > 200ms)
        pauses = intervals > 200
        features[1] = np.sum(pauses)
        if np.sum(pauses) > 0:
            features[2] = np.mean(intervals[pauses])
            features[3] = np.std(intervals[pauses]) if np.sum(pauses) > 1 else 0
        
        # Movement duration
        features[4] = (times[-1] - times[0]) / 1000.0
        
        # Idle ratio
        features[5] = np.sum(intervals[pauses]) / (np.sum(intervals) + 1e-6)
        
        # Burst detection (rapid movements)
        bursts = intervals < 20
        features[6] = np.sum(bursts)
        features[7] = np.sum(bursts) / (len(intervals) + 1e-6)
        
        return features
    
    def _extract_micropatterns(self, mouse_data):
        """Extract micro-movement patterns"""
        features = np.zeros(6)
        
        if len(mouse_data) < 5:
            return features
        
        positions = np.array([[p['x'], p['y']] for p in mouse_data])
        
        # Micro-corrections (small movements)
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        features[0] = np.sum(distances < 3)  # Count tiny movements
        
        # Direction reversals
        dx = np.diff(positions[:, 0])
        dy = np.diff(positions[:, 1])
        x_reversals = np.sum(dx[:-1] * dx[1:] < 0)
        y_reversals = np.sum(dy[:-1] * dy[1:] < 0)
        features[1] = x_reversals + y_reversals
        
        # Hesitation (slowing down)
        velocities = distances
        hesitations = np.sum((velocities[:-1] > 5) & (velocities[1:] < 2))
        features[2] = hesitations
        
        # Overshoots (reverse after fast movement)
        features[3] = np.sum((velocities[:-1] > 10) & (velocities[1:] < 3))
        
        # Tremor score (high frequency small oscillations)
        fft = np.abs(np.fft.fft(distances))
        high_freq_power = np.sum(fft[len(fft)//2:])
        total_power = np.sum(fft) + 1e-6
        features[4] = high_freq_power / total_power
        
        # Smoothness (low jerk)
        if len(distances) > 2:
            jerk = np.diff(np.diff(distances))
            features[5] = 1 / (np.mean(np.abs(jerk)) + 1)
        
        return features
    
    def _extract_keyboard(self, keyboard_data):
        """Extract keyboard timing features"""
        features = np.zeros(8)
        
        if len(keyboard_data) < 3:
            return features
        
        times = np.array([k['timestamp'] for k in keyboard_data])
        intervals = np.diff(times)
        
        if len(intervals) > 0:
            features[0] = np.mean(intervals)
            features[1] = np.std(intervals)
            features[2] = np.std(intervals) / (np.mean(intervals) + 1e-6)
        
        # Digraph latency (time between pairs of keys)
        if len(intervals) >= 2:
            digraphs = intervals[::2]  # Simplified
            features[3] = np.mean(digraphs)
            features[4] = np.std(digraphs)
        
        # Typing speed (chars per minute)
        duration = (times[-1] - times[0]) / 1000.0 / 60.0
        if duration > 0:
            features[5] = len(keyboard_data) / duration
        
        # Key hold time (if available)
        holds = [k.get('hold_time', 100) for k in keyboard_data]
        features[6] = np.mean(holds)
        features[7] = np.std(holds)
        
        return features
    
    def _extract_clicks(self, click_data):
        """Extract click pattern features"""
        features = np.zeros(4)
        
        if len(click_data) < 2:
            return features
        
        times = np.array([c['timestamp'] for c in click_data])
        intervals = np.diff(times)
        
        if len(intervals) > 0:
            features[0] = np.mean(intervals)
            features[1] = np.std(intervals)
            
            # Double-click ratio
            features[2] = np.sum(intervals < 300) / len(intervals)
        
        # Click precision (variance in position)
        if len(click_data) >= 2:
            positions = np.array([[c.get('x', 0), c.get('y', 0)] for c in click_data])
            features[3] = np.mean(np.std(positions, axis=0))
        
        return features
    
    def _extract_scroll(self, scroll_data):
        """Extract scroll pattern features"""
        features = np.zeros(4)
        
        if len(scroll_data) < 2:
            return features
        
        deltas = [s.get('deltaY', 0) for s in scroll_data]
        times = [s.get('timestamp', 0) for s in scroll_data]
        
        intervals = np.diff(times)
        intervals = np.maximum(intervals, 1)
        
        velocities = np.abs(np.array(deltas[1:])) / intervals
        
        if len(velocities) > 0:
            features[0] = np.mean(velocities)
            features[1] = np.std(velocities)
        
        # Direction changes
        signs = np.sign(deltas)
        features[2] = np.sum(np.diff(signs) != 0)
        
        # Burst ratio
        features[3] = np.sum(np.array(deltas) > 100) / (len(deltas) + 1e-6)
        
        return features
    
    def _extract_session(self, data):
        """Extract session-level features"""
        features = np.zeros(4)
        
        mouse_data = data.get('mouse_data', [])
        keyboard_data = data.get('keyboard_data', [])
        click_data = data.get('click_data', [])
        scroll_data = data.get('scroll_data', [])
        
        # Session duration
        all_times = []
        for m in mouse_data:
            all_times.append(m.get('timestamp', 0))
        for k in keyboard_data:
            all_times.append(k.get('timestamp', 0))
        
        if len(all_times) >= 2:
            features[0] = (max(all_times) - min(all_times)) / 1000.0
        
        # Activity density
        if features[0] > 0:
            total_events = len(mouse_data) + len(keyboard_data) + len(click_data) + len(scroll_data)
            features[1] = total_events / features[0]
        
        # Input diversity
        has_mouse = len(mouse_data) > 5
        has_keyboard = len(keyboard_data) > 3
        has_clicks = len(click_data) > 0
        has_scroll = len(scroll_data) > 0
        features[2] = sum([has_mouse, has_keyboard, has_clicks, has_scroll]) / 4
        
        # Behavioral consistency (low variance across time windows)
        # Simplified: just use overall activity variance
        features[3] = 0.5  # Placeholder
        
        return features


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

if TORCH_AVAILABLE:
    
    class BotDetectorNet(nn.Module):
        """
        Deep neural network for bot detection
        Architecture: Input -> Dense layers with BatchNorm + Dropout -> Output
        """
        
        def __init__(self, input_size=56, hidden_sizes=[128, 64, 32], dropout=0.3):
            super(BotDetectorNet, self).__init__()
            
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    
    class BehaviorDataset(Dataset):
        """Dataset for behavioral data"""
        
        def __init__(self, features, labels):
            self.features = torch.FloatTensor(features)
            self.labels = torch.FloatTensor(labels).unsqueeze(1)
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    
    class NeuralBotDetector:
        """
        Production neural network bot detector
        """
        
        def __init__(self, model_path=None):
            self.feature_extractor = AdvancedFeatureExtractor()
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Feature normalization stats
            self.feature_mean = None
            self.feature_std = None
            
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
        
        def train(self, training_data, labels, epochs=100, batch_size=32, learning_rate=0.001,
                  validation_split=0.2, early_stopping_patience=10):
            """
            Train the neural network
            
            Args:
                training_data: List of behavioral data dicts
                labels: List of labels (0=human, 1=bot)
                epochs: Training epochs
                batch_size: Batch size
                learning_rate: Learning rate
                validation_split: Fraction for validation
                early_stopping_patience: Stop if no improvement
            """
            print(f"[TRAIN] Extracting features from {len(training_data)} samples...")
            
            # Extract features
            features = np.array([self.feature_extractor.extract(d) for d in training_data])
            labels = np.array(labels)
            
            # Normalize features
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0) + 1e-6
            features = (features - self.feature_mean) / self.feature_std
            
            # Create dataset
            dataset = BehaviorDataset(features, labels)
            
            # Split into train/val
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model
            input_size = features.shape[1]
            self.model = BotDetectorNet(input_size=input_size).to(self.device)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            print(f"[TRAIN] Starting training for {epochs} epochs...")
            print(f"[TRAIN] Device: {self.device}")
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = self.model(batch_features)
                        loss = criterion(outputs, batch_labels)
                        val_loss += loss.item()
                        
                        predicted = (outputs > 0.5).float()
                        val_correct += (predicted == batch_labels).sum().item()
                        val_total += batch_labels.size(0)
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                scheduler.step(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"[TRAIN] Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"[TRAIN] Early stopping at epoch {epoch+1}")
                        break
            
            print(f"[TRAIN] Training complete. Best validation loss: {best_val_loss:.4f}")
            self._load_checkpoint()
        
        def predict(self, data):
            """
            Predict if behavior is bot-like
            
            Args:
                data: Behavioral data dict or pre-extracted features dict
            
            Returns:
                Prediction dict with bot_score, is_bot, confidence, etc.
            """
            # Check if data is already extracted features
            if isinstance(data, dict) and 'raw' in data:
                # Already extracted by legacy system
                features = self._convert_legacy_features(data)
            else:
                features = self.feature_extractor.extract(data)
            
            # Normalize
            if self.feature_mean is not None:
                features = (features - self.feature_mean) / self.feature_std
            
            # Predict
            if self.model is None:
                # Fallback to heuristic
                return self._heuristic_predict(features)
            
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                bot_probability = self.model(features_tensor).item()
            
            bot_score = bot_probability * 100
            is_bot = bot_score > 55
            
            # Calculate confidence
            confidence = abs(bot_probability - 0.5) * 2
            
            # Generate detection triggers (human-readable signals)
            triggers = self._generate_triggers(features, bot_score)
            
            return {
                'bot_score': round(bot_score, 1),
                'is_bot': is_bot,
                'confidence': round(confidence, 3),
                'verdict': self._get_verdict(bot_score),
                'model_type': 'neural_network',
                'feature_importance': self._get_feature_importance(features),
                'component_scores': self._get_component_scores(features),
                'triggers': triggers
            }
        
        def _convert_legacy_features(self, data):
            """Convert legacy feature dict to array"""
            raw = data.get('raw', {})
            # Map old features to new positions
            features = np.zeros(len(self.feature_extractor.feature_names))
            
            # Basic mapping (simplified)
            mapping = {
                'velocity_mean': 0, 'velocity_std': 1, 'velocity_cv': 3,
                'acceleration_mean': 4, 'acceleration_cv': 7,
                'path_straightness': 12, 'curvature_entropy': 16,
            }
            
            for key, idx in mapping.items():
                if key in raw:
                    features[idx] = raw[key]
            
            return features
        
        def _heuristic_predict(self, features):
            """Fallback heuristic prediction when no model is trained"""
            # Simple rule-based scoring
            score = 50.0
            
            # Velocity CV (low = bot)
            if features[3] < 0.3:
                score += 15
            elif features[3] > 0.6:
                score -= 15
            
            # Straightness (high = bot)
            if features[12] > 0.9:
                score += 20
            elif features[12] < 0.7:
                score -= 10
            
            # Curvature entropy (low = bot)
            if features[16] < 1.5:
                score += 15
            elif features[16] > 2.5:
                score -= 15
            
            score = max(0, min(100, score))
            
            return {
                'bot_score': round(score, 1),
                'is_bot': score > 55,
                'confidence': round(abs(score - 50) / 50, 3),
                'verdict': self._get_verdict(score),
                'model_type': 'heuristic_fallback',
                'component_scores': self._get_component_scores(features),
                'triggers': self._generate_triggers(features, score)
            }
        
        def _get_verdict(self, score):
            if score > 80:
                return "DEFINITELY_BOT"
            elif score > 60:
                return "LIKELY_BOT"
            elif score > 40:
                return "UNCERTAIN"
            elif score > 20:
                return "LIKELY_HUMAN"
            else:
                return "DEFINITELY_HUMAN"
        
        def _get_feature_importance(self, features):
            """Get which features contributed most to prediction"""
            # Simplified: return features with largest absolute values
            importance = {}
            for i, name in enumerate(self.feature_extractor.feature_names[:10]):
                importance[name] = float(abs(features[i]))
            return importance
        
        def _get_component_scores(self, features):
            """Calculate component scores for frontend display"""
            # Feature indices: velocity_cv=3, path_straightness=12, curvature_entropy=16, acceleration_cv=7
            # These are normalized 0-1 where 1 = human-like, 0 = bot-like
            
            # Velocity score: higher CV = more human-like (capped at 1.0)
            velocity_cv = features[3] if len(features) > 3 else 0.5
            velocity_score = min(1.0, velocity_cv / 0.8)  # 0.8+ CV = full score
            
            # Path score: lower straightness = more human-like
            straightness = features[12] if len(features) > 12 else 0.5
            path_score = 1.0 - straightness  # Invert: curved paths = higher score
            
            # Curvature score: higher entropy = more human-like
            entropy = features[16] if len(features) > 16 else 2.0
            curvature_score = min(1.0, entropy / 3.0)  # 3.0+ entropy = full score
            
            # Timing score: based on pause patterns and acceleration variance
            pause_ratio = features[15] if len(features) > 15 else 0.1
            acc_cv = features[7] if len(features) > 7 else 0.5
            timing_score = min(1.0, (pause_ratio * 10 + acc_cv) / 1.5)
            
            # Keystroke score: placeholder (would need keyboard data)
            keystroke_score = 0.5  # Neutral when no keyboard data
            
            return {
                'velocity': round(velocity_score, 3),
                'path': round(path_score, 3),
                'curvature': round(curvature_score, 3),
                'timing': round(timing_score, 3),
                'keystroke': round(keystroke_score, 3)
            }
        
        def _generate_triggers(self, features, bot_score):
            """Generate human-readable detection signals based on features"""
            triggers = []
            
            # Feature indices (from feature_names order)
            # velocity_mean=0, velocity_std=1, velocity_max=2, velocity_cv=3
            # acceleration_mean=4, acceleration_std=5, acceleration_max=6, acceleration_cv=7
            # angle_change_mean=8, angle_change_std=9, angle_change_max=10
            # direction_changes=11, path_straightness=12, path_efficiency=13
            # pause_count=14, pause_ratio=15, curvature_entropy=16
            
            try:
                # Velocity analysis
                velocity_cv = features[3] if len(features) > 3 else 0.5
                if velocity_cv < 0.2:
                    triggers.append({"type": "warning", "text": "Unnaturally consistent velocity"})
                elif velocity_cv > 1.5:
                    triggers.append({"type": "success", "text": "Natural velocity variation"})
                
                # Path straightness
                straightness = features[12] if len(features) > 12 else 0.5
                if straightness > 0.95:
                    triggers.append({"type": "warning", "text": "Perfectly straight mouse paths"})
                elif straightness < 0.7:
                    triggers.append({"type": "success", "text": "Natural curved movements"})
                
                # Direction changes
                dir_changes = features[11] if len(features) > 11 else 5
                if dir_changes < 2:
                    triggers.append({"type": "warning", "text": "Too few direction corrections"})
                elif dir_changes > 10:
                    triggers.append({"type": "success", "text": "Normal micro-corrections detected"})
                
                # Curvature entropy
                entropy = features[16] if len(features) > 16 else 2.0
                if entropy < 1.0:
                    triggers.append({"type": "warning", "text": "Low movement complexity"})
                elif entropy > 2.5:
                    triggers.append({"type": "success", "text": "High movement complexity"})
                
                # Pause patterns
                pause_ratio = features[15] if len(features) > 15 else 0.1
                if pause_ratio < 0.01:
                    triggers.append({"type": "warning", "text": "No natural pauses detected"})
                elif pause_ratio > 0.05:
                    triggers.append({"type": "success", "text": "Natural pause patterns"})
                
                # Acceleration patterns
                acc_cv = features[7] if len(features) > 7 else 0.5
                if acc_cv < 0.15:
                    triggers.append({"type": "warning", "text": "Robotic acceleration pattern"})
                
                # Overall score interpretation
                if bot_score > 80:
                    triggers.insert(0, {"type": "danger", "text": "HIGH BOT PROBABILITY"})
                elif bot_score > 60:
                    triggers.insert(0, {"type": "warning", "text": "Suspicious behavior detected"})
                elif bot_score < 30:
                    triggers.insert(0, {"type": "success", "text": "Human-like behavior confirmed"})
                
            except (IndexError, TypeError):
                triggers.append({"type": "info", "text": "Analysis in progress..."})
            
            return triggers[:6]  # Limit to 6 triggers
        
        def save_model(self, path):
            """Save trained model"""
            if self.model is None:
                print("[WARNING] No model to save")
                return
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'feature_mean': self.feature_mean,
                'feature_std': self.feature_std,
                'feature_names': self.feature_extractor.feature_names
            }, path)
            print(f"[SAVE] Model saved to {path}")
        
        def load_model(self, path):
            """Load trained model"""
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'feature_names' in checkpoint:
                input_size = len(checkpoint['feature_names'])
            elif 'input_size' in checkpoint:
                input_size = checkpoint['input_size']
            else:
                input_size = 56  # Default feature count
            
            self.model = BotDetectorNet(input_size=input_size).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.feature_mean = checkpoint.get('feature_mean', np.zeros(input_size))
            self.feature_std = checkpoint.get('feature_std', np.ones(input_size))
            
            print(f"[LOAD] Model loaded from {path}")
        
        def _save_checkpoint(self):
            """Save training checkpoint"""
            self._checkpoint = {
                'model_state_dict': self.model.state_dict(),
            }
        
        def _load_checkpoint(self):
            """Load training checkpoint"""
            if hasattr(self, '_checkpoint'):
                self.model.load_state_dict(self._checkpoint['model_state_dict'])


# ============================================================================
# DATA COLLECTION FOR REAL-WORLD TESTING
# ============================================================================

class DataCollector:
    """
    Collect and label behavioral data for training
    Use this to build your own dataset from real users
    """
    
    def __init__(self, output_dir="datasets/collected"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sessions = []
    
    def record_session(self, session_data, label, metadata=None):
        """
        Record a labeled session
        
        Args:
            session_data: Dict with mouse_data, keyboard_data, etc.
            label: 0=human, 1=bot
            metadata: Optional dict with user_id, timestamp, etc.
        """
        record = {
            'data': session_data,
            'label': label,
            'metadata': metadata or {}
        }
        
        self.sessions.append(record)
        
        # Auto-save every 100 sessions
        if len(self.sessions) % 100 == 0:
            self.save()
    
    def save(self):
        """Save collected data to disk"""
        path = os.path.join(self.output_dir, f"sessions_{len(self.sessions)}.json")
        with open(path, 'w') as f:
            json.dump(self.sessions, f)
        print(f"[SAVE] Saved {len(self.sessions)} sessions to {path}")
    
    def load_all(self):
        """Load all collected data"""
        all_sessions = []
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.json'):
                path = os.path.join(self.output_dir, filename)
                with open(path) as f:
                    all_sessions.extend(json.load(f))
        return all_sessions


# ============================================================================
# REAL-WORLD TESTING FRAMEWORK
# ============================================================================

class RealWorldTester:
    """
    Framework for testing model performance in production
    """
    
    def __init__(self, detector):
        self.detector = detector
        self.results = []
    
    def run_ab_test(self, sessions, ground_truth):
        """
        Run A/B test comparing predictions to ground truth
        
        Args:
            sessions: List of session data
            ground_truth: List of actual labels (0=human, 1=bot)
        """
        predictions = []
        
        for session in sessions:
            pred = self.detector.predict(session)
            predictions.append(1 if pred['is_bot'] else 0)
        
        # Calculate metrics
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 1)
        tn = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 0)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 1)
        
        accuracy = (tp + tn) / len(predictions)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # False positive rate (critical for user experience)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'total_samples': len(predictions)
        }
        
        print("\n" + "="*50)
        print("REAL-WORLD TEST RESULTS")
        print("="*50)
        print(f"Accuracy:           {accuracy:.1%}")
        print(f"Precision:          {precision:.1%}")
        print(f"Recall:             {recall:.1%}")
        print(f"F1 Score:           {f1:.3f}")
        print(f"False Positive Rate: {fpr:.1%} (lower is better)")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp}, FP: {fp}")
        print(f"  FN: {fn}, TN: {tn}")
        print("="*50)
        
        return results
    
    def shadow_mode(self, session, actual_label=None):
        """
        Run prediction in shadow mode (log but don't act)
        Use this for safe production testing
        """
        prediction = self.detector.predict(session)
        
        result = {
            'prediction': prediction,
            'actual': actual_label,
            'timestamp': np.datetime64('now')
        }
        
        self.results.append(result)
        
        return prediction
    
    def get_metrics(self):
        """Get accumulated metrics from shadow mode"""
        if not self.results:
            return {}
        
        correct = sum(1 for r in self.results 
                     if r['actual'] is not None 
                     and (r['prediction']['is_bot'] == (r['actual'] == 1)))
        
        total = sum(1 for r in self.results if r['actual'] is not None)
        
        return {
            'shadow_accuracy': correct / total if total > 0 else 0,
            'total_predictions': len(self.results),
            'labeled_predictions': total
        }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Network Bot Detector")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--dataset', type=str, default='synthetic', help='Dataset to use')
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        print("Please install PyTorch: pip install torch")
        exit(1)
    
    detector = NeuralBotDetector()
    
    if args.train:
        print("[INFO] Generating synthetic training data...")
        
        # Generate training data
        np.random.seed(42)
        training_data = []
        labels = []
        
        for _ in range(500):  # 500 human samples
            t = np.linspace(0, 4*np.pi, 50)
            data = {
                'mouse_data': [
                    {'x': float(200 + 100*np.sin(ti) + np.random.randn()*10),
                     'y': float(200 + 50*np.cos(ti*0.5) + np.random.randn()*10),
                     'timestamp': int(1000 + i*50 + np.random.randint(-30, 30))}
                    for i, ti in enumerate(t)
                ],
                'keyboard_data': [
                    {'key': 'a', 'timestamp': 1000 + i*100 + np.random.randint(-50, 50)}
                    for i in range(20)
                ],
                'click_data': [],
                'scroll_data': []
            }
            training_data.append(data)
            labels.append(0)  # Human
        
        for _ in range(500):  # 500 bot samples
            data = {
                'mouse_data': [
                    {'x': float(100 + i*10 + np.random.randn()*0.5),
                     'y': float(100 + i*10 + np.random.randn()*0.5),
                     'timestamp': int(1000 + i*50)}
                    for i in range(50)
                ],
                'keyboard_data': [
                    {'key': 'a', 'timestamp': 1000 + i*100}
                    for i in range(20)
                ],
                'click_data': [],
                'scroll_data': []
            }
            training_data.append(data)
            labels.append(1)  # Bot
        
        # Train
        detector.train(training_data, labels, epochs=50, batch_size=32)
        detector.save_model("models/neural_bot_detector.pth")
    
    if args.evaluate:
        # Load trained model for evaluation
        model_path = "models/neural_bot_detector.pth"
        if os.path.exists(model_path):
            detector.load_model(model_path)
        
        print("[INFO] Running evaluation...")
        tester = RealWorldTester(detector)
        
        # Generate test data
        np.random.seed(123)  # Different seed for test data
        test_data = []
        test_labels = []
        
        for _ in range(50):
            t = np.linspace(0, 4*np.pi, 50)
            data = {
                'mouse_data': [
                    {'x': float(200 + 100*np.sin(ti) + np.random.randn()*8),
                     'y': float(200 + 50*np.cos(ti*0.5) + np.random.randn()*8),
                     'timestamp': int(1000 + i*50 + np.random.randint(-25, 25))}
                    for i, ti in enumerate(t)
                ],
                'keyboard_data': [
                    {'key': 'a', 'timestamp': 1000 + i*100 + np.random.randint(-40, 40)}
                    for i in range(15)
                ],
                'click_data': [],
                'scroll_data': []
            }
            test_data.append(data)
            test_labels.append(0)
        
        for _ in range(50):
            data = {
                'mouse_data': [
                    {'x': float(100 + i*10 + np.random.randn()*0.3), 
                     'y': float(100 + i*10 + np.random.randn()*0.3), 
                     'timestamp': int(1000 + i*50)}
                    for i in range(50)
                ],
                'keyboard_data': [
                    {'key': 'a', 'timestamp': 1000 + i*100}
                    for i in range(15)
                ],
                'click_data': [],
                'scroll_data': []
            }
            test_data.append(data)
            test_labels.append(1)
        
        tester.run_ab_test(test_data, test_labels)
