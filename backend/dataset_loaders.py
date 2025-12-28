"""
Dataset Loaders for Real Training Data
Supports: Balabit Mouse Dynamics, CMU Keystroke
"""

import os
import csv
import json
import numpy as np
from glob import glob


class BalabitLoader:
    """
    Load Balabit Mouse Dynamics Challenge data
    
    Expected structure:
    datasets/balabit/
        training_files/
            user1/
                session_1.csv
                session_2.csv
            user2/
                ...
        test_files/
            ...
    
    CSV format: record timestamp,client timestamp,button,state,x,y
    """
    
    def __init__(self, base_path="datasets/balabit"):
        self.base_path = base_path
        self.training_path = os.path.join(base_path, "training_files")
        self.test_path = os.path.join(base_path, "test_files")
        
        # Alternative paths if structure is different
        if not os.path.exists(self.training_path):
            self.training_path = base_path
            self.test_path = base_path
    
    def load_session(self, csv_path):
        """Load a single session CSV file"""
        mouse_data = []
        click_data = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Handle different column name formats
                        x = float(row.get('x', row.get('X', 0)))
                        y = float(row.get('y', row.get('Y', 0)))
                        
                        # Timestamp handling
                        ts = row.get('client timestamp', row.get('timestamp', row.get('time', 0)))
                        timestamp = int(float(ts) * 1000) if float(ts) < 1e10 else int(float(ts))
                        
                        button = row.get('button', row.get('Button', 'NoButton'))
                        state = row.get('state', row.get('State', 'Move'))
                        
                        mouse_data.append({
                            'x': x,
                            'y': y,
                            'timestamp': timestamp
                        })
                        
                        # Track clicks
                        if button != 'NoButton' and state in ['Pressed', 'Down', 'pressed']:
                            click_data.append({
                                'x': x,
                                'y': y,
                                'timestamp': timestamp,
                                'button': button
                            })
                    except (ValueError, KeyError) as e:
                        continue
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return None
        
        if len(mouse_data) < 10:
            return None
        
        return {
            'mouse_data': mouse_data,
            'keyboard_data': [],
            'click_data': click_data,
            'scroll_data': []
        }
    
    def load_user_sessions(self, user_path, max_sessions=None):
        """Load all sessions for a user"""
        sessions = []
        
        # Get all files (Balabit files don't have .csv extension)
        all_files = []
        for f in os.listdir(user_path):
            file_path = os.path.join(user_path, f)
            if os.path.isfile(file_path):
                all_files.append(file_path)
        
        if max_sessions:
            all_files = all_files[:max_sessions]
        
        for file_path in all_files:
            session = self.load_session(file_path)
            if session:
                sessions.append(session)
        
        return sessions
    
    def load_all(self, max_users=None, max_sessions_per_user=50):
        """Load all training data"""
        all_sessions = []
        labels = []  # All human data, label = 0
        
        # Find user directories
        if os.path.exists(self.training_path):
            user_dirs = [d for d in os.listdir(self.training_path) 
                        if os.path.isdir(os.path.join(self.training_path, d))]
        else:
            print(f"Training path not found: {self.training_path}")
            return [], []
        
        if max_users:
            user_dirs = user_dirs[:max_users]
        
        print(f"[BALABIT] Found {len(user_dirs)} users")
        
        for user_dir in user_dirs:
            user_path = os.path.join(self.training_path, user_dir)
            sessions = self.load_user_sessions(user_path, max_sessions_per_user)
            
            print(f"  - {user_dir}: {len(sessions)} sessions")
            
            all_sessions.extend(sessions)
            labels.extend([0] * len(sessions))  # Human = 0
        
        print(f"[BALABIT] Total: {len(all_sessions)} sessions loaded")
        return all_sessions, labels
    
    def generate_bot_samples(self, human_sessions, num_samples=None):
        """
        Generate synthetic bot samples based on human sessions
        This creates straight-line, constant-velocity versions
        """
        if num_samples is None:
            num_samples = len(human_sessions)
        
        bot_sessions = []
        
        for i in range(num_samples):
            # Get a random human session as template
            template = human_sessions[i % len(human_sessions)]
            mouse_data = template['mouse_data']
            
            if len(mouse_data) < 5:
                continue
            
            # Create bot-like movement (straight lines, constant timing)
            start = mouse_data[0]
            end = mouse_data[-1]
            
            num_points = len(mouse_data)
            
            # Perfectly straight line with constant timing
            bot_mouse = []
            for j in range(num_points):
                t = j / (num_points - 1)
                bot_mouse.append({
                    'x': start['x'] + t * (end['x'] - start['x']) + np.random.randn() * 0.5,
                    'y': start['y'] + t * (end['y'] - start['y']) + np.random.randn() * 0.5,
                    'timestamp': start['timestamp'] + j * 20  # Constant 20ms intervals
                })
            
            bot_sessions.append({
                'mouse_data': bot_mouse,
                'keyboard_data': [],
                'click_data': [],
                'scroll_data': []
            })
        
        return bot_sessions


class CMUKeystrokeLoader:
    """
    Load CMU Keystroke Dynamics Dataset (DSL-StrongPasswordData.csv)
    
    Format: subject,sessionIndex,rep,H.period,DD.period.t,UD.period.t,...
    
    H.* = Hold time (key down to key up)
    DD.* = Down-Down time (key down to next key down)
    UD.* = Up-Down time (key up to next key down)
    """
    
    def __init__(self, csv_path="datasets/DSL-StrongPasswordData.csv"):
        self.csv_path = csv_path
        self.password = ".tie5Roanl"  # The password used in the study
    
    def load_all(self, max_subjects=None, max_reps_per_subject=50):
        """Load keystroke data and convert to our format"""
        sessions = []
        labels = []
        
        if not os.path.exists(self.csv_path):
            print(f"[CMU] File not found: {self.csv_path}")
            return [], []
        
        subjects_data = {}
        
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                subject = row['subject']
                
                if subject not in subjects_data:
                    subjects_data[subject] = []
                
                if max_reps_per_subject and len(subjects_data[subject]) >= max_reps_per_subject:
                    continue
                
                # Extract timing data
                session = self._parse_keystroke_row(row)
                if session:
                    subjects_data[subject].append(session)
        
        # Flatten and create labels
        subjects = list(subjects_data.keys())
        if max_subjects:
            subjects = subjects[:max_subjects]
        
        print(f"[CMU] Found {len(subjects)} subjects")
        
        for subject in subjects:
            subject_sessions = subjects_data[subject]
            print(f"  - {subject}: {len(subject_sessions)} sessions")
            
            sessions.extend(subject_sessions)
            labels.extend([0] * len(subject_sessions))  # Human = 0
        
        print(f"[CMU] Total: {len(sessions)} sessions loaded")
        return sessions, labels
    
    def _parse_keystroke_row(self, row):
        """Convert a CSV row to our session format"""
        keyboard_data = []
        base_time = 1000  # Start at 1 second
        
        # Keys in ".tie5Roanl"
        keys = list(self.password)
        
        current_time = base_time
        
        for i, key in enumerate(keys):
            # Get hold time for this key
            hold_col = f'H.{key}' if i == 0 else f'H.{keys[i-1]}.{key}' if i > 0 else f'H.{key}'
            
            # Try different column name formats
            hold_time = 0.1  # Default 100ms
            for col_name in [f'H.{key}', f'H.period' if key == '.' else None]:
                if col_name and col_name in row:
                    try:
                        hold_time = float(row[col_name])
                        break
                    except:
                        pass
            
            # Get DD time (inter-key interval)
            if i > 0:
                dd_col = f'DD.{keys[i-1]}.{key}'
                if dd_col in row:
                    try:
                        dd_time = float(row[dd_col])
                        current_time += int(dd_time * 1000)
                    except:
                        current_time += 100  # Default 100ms
                else:
                    current_time += 100
            
            keyboard_data.append({
                'key': key,
                'timestamp': current_time,
                'hold_time': int(hold_time * 1000)
            })
        
        if len(keyboard_data) < 5:
            return None
        
        return {
            'mouse_data': [],
            'keyboard_data': keyboard_data,
            'click_data': [],
            'scroll_data': []
        }
    
    def generate_bot_samples(self, human_sessions, num_samples=None):
        """Generate synthetic bot keystroke patterns"""
        if num_samples is None:
            num_samples = len(human_sessions)
        
        bot_sessions = []
        
        for i in range(num_samples):
            # Perfectly regular typing
            keyboard_data = []
            base_time = 1000
            
            for j, key in enumerate(self.password):
                keyboard_data.append({
                    'key': key,
                    'timestamp': base_time + j * 100,  # Exactly 100ms between keys
                    'hold_time': 50  # Exactly 50ms hold
                })
            
            bot_sessions.append({
                'mouse_data': [],
                'keyboard_data': keyboard_data,
                'click_data': [],
                'scroll_data': []
            })
        
        return bot_sessions


def load_and_prepare_training_data(
    balabit_path="datasets/balabit",
    cmu_path="datasets/DSL-StrongPasswordData.csv",
    max_samples_per_source=500
):
    """
    Load both datasets and prepare balanced training data
    """
    all_sessions = []
    all_labels = []
    
    # Load Balabit mouse data
    print("\n" + "="*50)
    print("LOADING BALABIT MOUSE DATA")
    print("="*50)
    
    balabit = BalabitLoader(balabit_path)
    mouse_sessions, mouse_labels = balabit.load_all(max_sessions_per_user=100)
    
    if mouse_sessions:
        # Limit to max samples
        if len(mouse_sessions) > max_samples_per_source:
            indices = np.random.choice(len(mouse_sessions), max_samples_per_source, replace=False)
            mouse_sessions = [mouse_sessions[i] for i in indices]
            mouse_labels = [mouse_labels[i] for i in indices]
        
        all_sessions.extend(mouse_sessions)
        all_labels.extend(mouse_labels)
        
        # Generate matching bot samples
        bot_mouse = balabit.generate_bot_samples(mouse_sessions)
        all_sessions.extend(bot_mouse)
        all_labels.extend([1] * len(bot_mouse))
    
    # Load CMU keystroke data
    print("\n" + "="*50)
    print("LOADING CMU KEYSTROKE DATA")
    print("="*50)
    
    cmu = CMUKeystrokeLoader(cmu_path)
    key_sessions, key_labels = cmu.load_all(max_reps_per_subject=50)
    
    if key_sessions:
        # Limit to max samples
        if len(key_sessions) > max_samples_per_source:
            indices = np.random.choice(len(key_sessions), max_samples_per_source, replace=False)
            key_sessions = [key_sessions[i] for i in indices]
            key_labels = [key_labels[i] for i in indices]
        
        all_sessions.extend(key_sessions)
        all_labels.extend(key_labels)
        
        # Generate matching bot samples
        bot_keys = cmu.generate_bot_samples(key_sessions)
        all_sessions.extend(bot_keys)
        all_labels.extend([1] * len(bot_keys))
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total sessions: {len(all_sessions)}")
    print(f"Human samples: {sum(1 for l in all_labels if l == 0)}")
    print(f"Bot samples: {sum(1 for l in all_labels if l == 1)}")
    
    return all_sessions, all_labels


if __name__ == "__main__":
    # Test loading
    sessions, labels = load_and_prepare_training_data(
        balabit_path="datasets/balabit",  # Correct path
        cmu_path="datasets/DSL-StrongPasswordData.csv"
    )
    
    if sessions:
        print("\nSample human session:")
        human_idx = labels.index(0)
        print(f"  Mouse points: {len(sessions[human_idx]['mouse_data'])}")
        print(f"  Keyboard events: {len(sessions[human_idx]['keyboard_data'])}")
        
        print("\nSample bot session:")
        bot_idx = labels.index(1)
        print(f"  Mouse points: {len(sessions[bot_idx]['mouse_data'])}")
        print(f"  Keyboard events: {len(sessions[bot_idx]['keyboard_data'])}")
