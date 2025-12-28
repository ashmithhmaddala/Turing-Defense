"""
Production Data Collection Pipeline
=====================================
Continuously collects real attack samples for model improvement.

Features:
- Real-time data collection from production
- Automated labeling via honeypots
- Attack sample detection and logging
- Model retraining triggers
- A/B testing infrastructure
"""

import os
import json
import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import random


class SampleSource(Enum):
    """Source of collected samples"""
    PRODUCTION = "production"      # Real user traffic
    HONEYPOT = "honeypot"          # Fake targets that only bots hit
    CHALLENGE = "challenge"         # Challenge mode submissions
    REPORTED = "reported"           # User-reported bots
    SYNTHETIC = "synthetic"         # Generated samples


class LabelConfidence(Enum):
    """How confident we are in the label"""
    VERIFIED = "verified"           # Ground truth (honeypot, challenge)
    HIGH = "high"                   # Multiple signals agree
    MEDIUM = "medium"               # Some signals
    LOW = "low"                     # Single signal
    UNKNOWN = "unknown"             # Needs review


@dataclass
class CollectedSample:
    """A collected behavioral sample with metadata"""
    sample_id: str
    timestamp: str
    session_data: Dict
    
    # Labeling
    label: int  # 0=human, 1=bot
    confidence: str
    source: str
    
    # Detection info
    model_prediction: float
    model_version: str
    detection_signals: List[str]
    
    # Metadata
    user_agent: str
    ip_hash: str  # Hashed for privacy
    session_duration_ms: int
    
    # Review status
    reviewed: bool = False
    reviewer_label: Optional[int] = None


class HoneypotManager:
    """
    Manages honeypot traps that only bots would trigger.
    These provide ground-truth bot labels.
    """
    
    def __init__(self):
        self.honeypots = {}
        self.triggered_sessions = set()
        
    def create_honeypot(
        self,
        honeypot_id: str,
        honeypot_type: str,
        config: Dict
    ) -> Dict:
        """
        Create a new honeypot trap.
        
        Types:
        - hidden_field: Invisible form field (bots fill it, humans don't)
        - timing_trap: Action too fast to be human
        - invisible_link: CSS-hidden link (bots click, humans don't see)
        - fake_login: Fake login form on non-existent page
        - rapid_form: Form submitted impossibly fast
        """
        honeypot = {
            "id": honeypot_id,
            "type": honeypot_type,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "triggers": 0,
            "sessions": []
        }
        self.honeypots[honeypot_id] = honeypot
        return honeypot
    
    def check_trigger(
        self,
        honeypot_id: str,
        session_id: str,
        trigger_data: Dict
    ) -> bool:
        """Check if a session triggered a honeypot"""
        
        honeypot = self.honeypots.get(honeypot_id)
        if not honeypot:
            return False
        
        triggered = False
        htype = honeypot["type"]
        config = honeypot["config"]
        
        if htype == "hidden_field":
            # Bot filled an invisible field
            field_value = trigger_data.get("field_value", "")
            if field_value:  # Should be empty
                triggered = True
                
        elif htype == "timing_trap":
            # Action completed too fast
            action_time = trigger_data.get("action_time_ms", float('inf'))
            min_time = config.get("min_time_ms", 500)
            if action_time < min_time:
                triggered = True
                
        elif htype == "invisible_link":
            # Clicked on invisible element
            if trigger_data.get("clicked"):
                triggered = True
                
        elif htype == "rapid_form":
            # Form submitted too quickly
            time_to_submit = trigger_data.get("time_to_submit_ms", float('inf'))
            min_time = config.get("min_time_ms", 2000)
            if time_to_submit < min_time:
                triggered = True
        
        if triggered:
            honeypot["triggers"] += 1
            honeypot["sessions"].append({
                "session_id": session_id,
                "triggered_at": datetime.now().isoformat(),
                "trigger_data": trigger_data
            })
            self.triggered_sessions.add(session_id)
        
        return triggered
    
    def is_known_bot(self, session_id: str) -> bool:
        """Check if session is a known bot (triggered honeypot)"""
        return session_id in self.triggered_sessions
    
    def get_stats(self) -> Dict:
        """Get honeypot statistics"""
        return {
            "total_honeypots": len(self.honeypots),
            "total_triggers": sum(h["triggers"] for h in self.honeypots.values()),
            "known_bots": len(self.triggered_sessions),
            "honeypots": {
                hid: {
                    "type": h["type"],
                    "triggers": h["triggers"]
                }
                for hid, h in self.honeypots.items()
            }
        }


class DataCollector:
    """
    Production data collection system.
    Collects behavioral data and labels it automatically when possible.
    """
    
    def __init__(
        self,
        storage_path: str = "collected_data",
        model_predictor: Optional[Callable] = None
    ):
        self.storage_path = storage_path
        self.model_predictor = model_predictor
        self.model_version = "unknown"
        
        self.honeypot_manager = HoneypotManager()
        self.samples: List[CollectedSample] = []
        self.session_buffer: Dict[str, Dict] = {}
        
        # Statistics
        self.stats = {
            "total_collected": 0,
            "humans": 0,
            "bots": 0,
            "unknown": 0,
            "by_source": defaultdict(int),
            "by_confidence": defaultdict(int)
        }
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Setup default honeypots
        self._setup_default_honeypots()
    
    def _setup_default_honeypots(self):
        """Create standard honeypot traps"""
        
        # Hidden form field
        self.honeypot_manager.create_honeypot(
            "hidden_email",
            "hidden_field",
            {"field_name": "email_confirm", "css": "display:none"}
        )
        
        # Timing trap
        self.honeypot_manager.create_honeypot(
            "fast_submit",
            "timing_trap",
            {"min_time_ms": 500, "description": "Form submit under 500ms"}
        )
        
        # Rapid form completion
        self.honeypot_manager.create_honeypot(
            "rapid_form",
            "rapid_form",
            {"min_time_ms": 2000, "description": "Full form under 2 seconds"}
        )
        
        # Invisible link
        self.honeypot_manager.create_honeypot(
            "hidden_link",
            "invisible_link",
            {"css": "position:absolute; left:-9999px"}
        )
    
    def start_session(
        self,
        session_id: str,
        user_agent: str = "",
        ip_address: str = ""
    ) -> Dict:
        """Start tracking a new session"""
        
        # Hash IP for privacy
        ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16]
        
        self.session_buffer[session_id] = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "start_timestamp": time.time() * 1000,
            "user_agent": user_agent,
            "ip_hash": ip_hash,
            "mouse_movements": [],
            "clicks": [],
            "keystrokes": [],
            "scrolls": [],
            "honeypot_triggers": [],
            "detection_signals": []
        }
        
        return {"status": "started", "session_id": session_id}
    
    def add_event(
        self,
        session_id: str,
        event_type: str,
        event_data: Dict
    ):
        """Add behavioral event to session"""
        
        if session_id not in self.session_buffer:
            self.start_session(session_id)
        
        session = self.session_buffer[session_id]
        
        if event_type == "mouse":
            session["mouse_movements"].append(event_data)
        elif event_type == "click":
            session["clicks"].append(event_data)
        elif event_type == "keydown" or event_type == "keyup":
            session["keystrokes"].append({**event_data, "type": event_type})
        elif event_type == "scroll":
            session["scrolls"].append(event_data)
    
    def check_honeypot(
        self,
        session_id: str,
        honeypot_id: str,
        trigger_data: Dict
    ) -> bool:
        """Check if session triggered a honeypot"""
        
        triggered = self.honeypot_manager.check_trigger(
            honeypot_id, session_id, trigger_data
        )
        
        if triggered and session_id in self.session_buffer:
            self.session_buffer[session_id]["honeypot_triggers"].append({
                "honeypot_id": honeypot_id,
                "timestamp": datetime.now().isoformat()
            })
            self.session_buffer[session_id]["detection_signals"].append(
                f"honeypot:{honeypot_id}"
            )
        
        return triggered
    
    def end_session(
        self,
        session_id: str,
        source: SampleSource = SampleSource.PRODUCTION,
        known_label: Optional[int] = None
    ) -> Optional[CollectedSample]:
        """
        End session and create collected sample.
        
        Args:
            session_id: Session to end
            source: Where this sample came from
            known_label: Ground truth label if known (0=human, 1=bot)
            
        Returns:
            CollectedSample or None if session not found
        """
        
        if session_id not in self.session_buffer:
            return None
        
        session = self.session_buffer.pop(session_id)
        
        # Calculate duration
        duration = time.time() * 1000 - session["start_timestamp"]
        
        # Create session data for model
        session_data = {
            "mouse_movements": session["mouse_movements"],
            "clicks": session["clicks"],
            "keystrokes": session["keystrokes"],
            "scrolls": session.get("scrolls", [])
        }
        
        # Get model prediction
        prediction = 0.5
        if self.model_predictor:
            try:
                result = self.model_predictor(session_data)
                prediction = result.get("confidence", 0.5)
            except Exception as e:
                print(f"Prediction error: {e}")
        
        # Determine label and confidence
        label, confidence = self._determine_label(
            session_id, session, prediction, known_label
        )
        
        # Create sample
        sample = CollectedSample(
            sample_id=f"{session_id}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            session_data=session_data,
            label=label,
            confidence=confidence.value,
            source=source.value,
            model_prediction=prediction,
            model_version=self.model_version,
            detection_signals=session["detection_signals"],
            user_agent=session["user_agent"],
            ip_hash=session["ip_hash"],
            session_duration_ms=int(duration)
        )
        
        self.samples.append(sample)
        
        # Update stats
        self.stats["total_collected"] += 1
        self.stats["by_source"][source.value] += 1
        self.stats["by_confidence"][confidence.value] += 1
        
        if label == 0:
            self.stats["humans"] += 1
        elif label == 1:
            self.stats["bots"] += 1
        else:
            self.stats["unknown"] += 1
        
        return sample
    
    def _determine_label(
        self,
        session_id: str,
        session: Dict,
        prediction: float,
        known_label: Optional[int]
    ) -> tuple:
        """Determine sample label and confidence"""
        
        # Ground truth from known label
        if known_label is not None:
            return known_label, LabelConfidence.VERIFIED
        
        # Honeypot triggered = definitely bot
        if self.honeypot_manager.is_known_bot(session_id):
            return 1, LabelConfidence.VERIFIED
        
        # Strong model prediction
        if prediction > 0.95:
            return 1, LabelConfidence.HIGH
        elif prediction < 0.05:
            return 0, LabelConfidence.HIGH
        elif prediction > 0.8:
            return 1, LabelConfidence.MEDIUM
        elif prediction < 0.2:
            return 0, LabelConfidence.MEDIUM
        elif prediction > 0.6:
            return 1, LabelConfidence.LOW
        elif prediction < 0.4:
            return 0, LabelConfidence.LOW
        else:
            return -1, LabelConfidence.UNKNOWN
    
    def save_samples(self, filename: Optional[str] = None) -> str:
        """Save collected samples to disk"""
        
        if filename is None:
            filename = f"samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.storage_path, filename)
        
        data = {
            "collected_at": datetime.now().isoformat(),
            "model_version": self.model_version,
            "stats": dict(self.stats),
            "honeypot_stats": self.honeypot_manager.get_stats(),
            "samples": [asdict(s) for s in self.samples]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Saved {len(self.samples)} samples to {filepath}")
        return filepath
    
    def load_samples(self, filepath: str) -> List[CollectedSample]:
        """Load samples from disk"""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        samples = []
        for s in data.get("samples", []):
            samples.append(CollectedSample(**s))
        
        return samples
    
    def get_training_data(
        self,
        min_confidence: LabelConfidence = LabelConfidence.MEDIUM,
        label_filter: Optional[int] = None
    ) -> tuple:
        """
        Get samples suitable for training.
        
        Args:
            min_confidence: Minimum confidence level
            label_filter: Only return specific label (0 or 1)
            
        Returns:
            (sessions, labels) tuple
        """
        
        confidence_order = [
            LabelConfidence.VERIFIED,
            LabelConfidence.HIGH,
            LabelConfidence.MEDIUM,
            LabelConfidence.LOW,
            LabelConfidence.UNKNOWN
        ]
        min_idx = confidence_order.index(min_confidence)
        allowed = set(c.value for c in confidence_order[:min_idx + 1])
        
        sessions = []
        labels = []
        
        for sample in self.samples:
            if sample.confidence not in allowed:
                continue
            if sample.label < 0:  # Unknown
                continue
            if label_filter is not None and sample.label != label_filter:
                continue
            
            sessions.append(sample.session_data)
            labels.append(sample.label)
        
        return sessions, labels
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            **self.stats,
            "honeypots": self.honeypot_manager.get_stats(),
            "pending_sessions": len(self.session_buffer)
        }


class RetrainingTrigger:
    """
    Monitors collected data and triggers model retraining when needed.
    """
    
    def __init__(
        self,
        collector: DataCollector,
        retrain_callback: Callable,
        min_samples: int = 100,
        accuracy_threshold: float = 0.95,
        check_interval_hours: int = 24
    ):
        self.collector = collector
        self.retrain_callback = retrain_callback
        self.min_samples = min_samples
        self.accuracy_threshold = accuracy_threshold
        self.check_interval = check_interval_hours * 3600
        
        self.last_check = time.time()
        self.last_retrain = time.time()
        self.retrain_count = 0
        
    def should_retrain(self) -> tuple:
        """
        Check if retraining is needed.
        
        Returns:
            (should_retrain, reason)
        """
        
        stats = self.collector.get_stats()
        
        # Not enough new samples
        new_samples = stats["total_collected"]
        if new_samples < self.min_samples:
            return False, f"Not enough samples ({new_samples}/{self.min_samples})"
        
        # Calculate model accuracy on new samples
        sessions, labels = self.collector.get_training_data(
            min_confidence=LabelConfidence.HIGH
        )
        
        if len(sessions) < 50:
            return False, "Not enough high-confidence samples"
        
        # Check accuracy
        if self.collector.model_predictor:
            correct = 0
            for session, label in zip(sessions, labels):
                try:
                    pred = self.collector.model_predictor(session)
                    pred_label = 1 if pred.get("confidence", 0) > 0.5 else 0
                    if pred_label == label:
                        correct += 1
                except:
                    pass
            
            accuracy = correct / len(sessions)
            
            if accuracy < self.accuracy_threshold:
                return True, f"Accuracy dropped to {accuracy:.1%}"
        
        # Time-based retrain
        hours_since_retrain = (time.time() - self.last_retrain) / 3600
        if hours_since_retrain > 168:  # 1 week
            return True, f"Weekly retraining ({hours_since_retrain:.0f}h since last)"
        
        return False, "Model performing well"
    
    def check_and_retrain(self) -> Optional[Dict]:
        """Check if retraining needed and execute if so"""
        
        should, reason = self.should_retrain()
        
        if not should:
            return {"retrained": False, "reason": reason}
        
        print(f"[RETRAIN TRIGGER] {reason}")
        
        # Get training data
        sessions, labels = self.collector.get_training_data()
        
        # Execute retraining
        result = self.retrain_callback(sessions, labels)
        
        self.last_retrain = time.time()
        self.retrain_count += 1
        
        # Clear used samples
        self.collector.samples = []
        
        return {
            "retrained": True,
            "reason": reason,
            "samples_used": len(sessions),
            "result": result
        }


class ABTestingFramework:
    """
    A/B testing for comparing model versions in production.
    """
    
    def __init__(self, models: Dict[str, Callable]):
        """
        Args:
            models: Dictionary of model_name -> predictor_function
        """
        self.models = models
        self.results = {name: {
            "predictions": 0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            "latencies": []
        } for name in models}
        
        self.traffic_split = {name: 1.0 / len(models) for name in models}
        
    def predict(self, session_data: Dict) -> tuple:
        """
        Make prediction using randomly selected model.
        
        Returns:
            (model_name, prediction_result)
        """
        # Select model based on traffic split
        r = random.random()
        cumulative = 0
        selected_model = list(self.models.keys())[0]
        
        for name, split in self.traffic_split.items():
            cumulative += split
            if r < cumulative:
                selected_model = name
                break
        
        # Make prediction
        start_time = time.time()
        result = self.models[selected_model](session_data)
        latency = (time.time() - start_time) * 1000
        
        # Record metrics
        self.results[selected_model]["predictions"] += 1
        self.results[selected_model]["latencies"].append(latency)
        
        return selected_model, result
    
    def record_outcome(
        self,
        model_name: str,
        predicted_bot: bool,
        actual_bot: bool
    ):
        """Record ground truth for a prediction"""
        
        if model_name not in self.results:
            return
        
        results = self.results[model_name]
        
        if predicted_bot and actual_bot:
            results["true_positives"] += 1
        elif predicted_bot and not actual_bot:
            results["false_positives"] += 1
        elif not predicted_bot and actual_bot:
            results["false_negatives"] += 1
        else:
            results["true_negatives"] += 1
    
    def get_results(self) -> Dict:
        """Get A/B test results"""
        
        summary = {}
        
        for name, results in self.results.items():
            tp = results["true_positives"]
            fp = results["false_positives"]
            tn = results["true_negatives"]
            fn = results["false_negatives"]
            
            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            avg_latency = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0
            
            summary[name] = {
                "predictions": results["predictions"],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "avg_latency_ms": avg_latency,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0
            }
        
        return summary
    
    def set_traffic_split(self, splits: Dict[str, float]):
        """Set traffic split between models (should sum to 1.0)"""
        total = sum(splits.values())
        self.traffic_split = {k: v / total for k, v in splits.items()}


if __name__ == "__main__":
    print("Production Data Collection Pipeline")
    print("=" * 60)
    
    # Demo usage
    collector = DataCollector(storage_path="demo_collected_data")
    
    # Simulate collecting some sessions
    for i in range(10):
        session_id = f"session_{i}"
        collector.start_session(session_id, "Mozilla/5.0", f"192.168.1.{i}")
        
        # Add some events
        for j in range(50):
            collector.add_event(session_id, "mouse", {
                "x": random.randint(0, 1920),
                "y": random.randint(0, 1080),
                "timestamp": j * 16
            })
        
        # Some trigger honeypots
        if i % 3 == 0:
            collector.check_honeypot(session_id, "fast_submit", {"action_time_ms": 100})
        
        # End session
        source = SampleSource.HONEYPOT if i % 3 == 0 else SampleSource.PRODUCTION
        collector.end_session(session_id, source=source)
    
    # Print stats
    print("\nCollection Stats:")
    stats = collector.get_stats()
    print(f"  Total collected: {stats['total_collected']}")
    print(f"  Humans: {stats['humans']}")
    print(f"  Bots: {stats['bots']}")
    print(f"  By source: {dict(stats['by_source'])}")
    print(f"  By confidence: {dict(stats['by_confidence'])}")
    print(f"\nHoneypot Stats:")
    print(f"  {stats['honeypots']}")
