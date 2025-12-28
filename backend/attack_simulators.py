"""
Attack Simulators - Mimics Real Bot Farm & Attack Tool Patterns
================================================================
Simulates behavioral patterns from real automation tools:
- Selenium/WebDriver bots
- Puppeteer/Playwright headless browsers  
- AutoHotkey/AutoIt macro tools
- Python automation (pyautogui, pynput)
- Credential stuffing tools
- Click farms (human-assisted bots)
- Mobile emulators
- AI-powered evasion bots

For training robust bot detection models.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class BotType(Enum):
    """Categories of bots based on sophistication"""
    NAIVE = "naive"                    # Simple scripts, easy to detect
    SELENIUM = "selenium"              # WebDriver-based automation
    PUPPETEER = "puppeteer"            # Headless browser automation
    MACRO = "macro"                    # AutoHotkey/AutoIt tools
    CREDENTIAL_STUFFER = "credential"  # Login attack tools
    CLICK_FARM = "click_farm"          # Human-assisted bots
    MOBILE_EMU = "mobile_emulator"     # Mobile device emulation
    EVASIVE = "evasive"                # AI-powered evasion attempts
    ADVANCED_EVASIVE = "advanced"      # State-of-the-art evasion


@dataclass
class BotProfile:
    """Configuration for a specific bot type"""
    name: str
    bot_type: BotType
    description: str
    
    # Mouse behavior parameters
    mouse_speed_range: Tuple[float, float]  # pixels per ms
    mouse_acceleration: float  # 0 = constant, 1 = human-like
    mouse_jitter: float  # random noise in movement
    mouse_curve_factor: float  # 0 = straight lines, 1 = bezier curves
    click_precision: float  # how close to target center
    double_click_speed: float  # ms between double clicks
    
    # Timing parameters  
    action_interval_range: Tuple[int, int]  # ms between actions
    typing_speed_range: Tuple[int, int]  # ms per character
    typing_variance: float  # timing randomness
    hesitation_probability: float  # chance of pause before action
    
    # Behavioral quirks
    scroll_pattern: str  # "instant", "smooth", "stepped", "variable"
    focus_changes: bool  # simulates human multitasking
    micro_corrections: bool  # small mouse adjustments
    fatigue_simulation: bool  # behavior changes over time
    
    # Detection evasion
    randomization_level: float  # 0-1, how much timing varies
    human_noise_injection: float  # 0-1, noise to seem human


# ============================================================
# BOT PROFILES - Based on Real Tool Behaviors
# ============================================================

BOT_PROFILES = {
    # Level 1: Naive bots - Easy to detect
    "naive_script": BotProfile(
        name="Naive Python Script",
        bot_type=BotType.NAIVE,
        description="Basic automation with no evasion attempts",
        mouse_speed_range=(5.0, 5.0),  # Constant speed
        mouse_acceleration=0.0,
        mouse_jitter=0.0,
        mouse_curve_factor=0.0,  # Straight lines only
        click_precision=1.0,  # Perfect clicks
        double_click_speed=50,  # Inhumanly fast
        action_interval_range=(100, 100),  # Fixed timing
        typing_speed_range=(10, 10),  # Constant typing
        typing_variance=0.0,
        hesitation_probability=0.0,
        scroll_pattern="instant",
        focus_changes=False,
        micro_corrections=False,
        fatigue_simulation=False,
        randomization_level=0.0,
        human_noise_injection=0.0
    ),
    
    # Level 2: Selenium WebDriver
    "selenium_basic": BotProfile(
        name="Selenium WebDriver (Basic)",
        bot_type=BotType.SELENIUM,
        description="Standard Selenium with ActionChains",
        mouse_speed_range=(2.0, 4.0),
        mouse_acceleration=0.1,
        mouse_jitter=0.5,
        mouse_curve_factor=0.1,  # Slight curves
        click_precision=0.95,
        double_click_speed=100,
        action_interval_range=(50, 200),
        typing_speed_range=(20, 50),
        typing_variance=0.1,
        hesitation_probability=0.0,
        scroll_pattern="instant",
        focus_changes=False,
        micro_corrections=False,
        fatigue_simulation=False,
        randomization_level=0.1,
        human_noise_injection=0.0
    ),
    
    "selenium_undetected": BotProfile(
        name="undetected-chromedriver",
        bot_type=BotType.SELENIUM,
        description="Selenium with undetected-chromedriver patches",
        mouse_speed_range=(1.0, 3.0),
        mouse_acceleration=0.3,
        mouse_jitter=1.0,
        mouse_curve_factor=0.3,
        click_precision=0.9,
        double_click_speed=150,
        action_interval_range=(100, 500),
        typing_speed_range=(30, 80),
        typing_variance=0.2,
        hesitation_probability=0.1,
        scroll_pattern="stepped",
        focus_changes=False,
        micro_corrections=True,
        fatigue_simulation=False,
        randomization_level=0.3,
        human_noise_injection=0.1
    ),
    
    # Level 3: Puppeteer/Playwright
    "puppeteer_basic": BotProfile(
        name="Puppeteer (Basic)",
        bot_type=BotType.PUPPETEER,
        description="Headless Chrome with Puppeteer",
        mouse_speed_range=(1.5, 3.5),
        mouse_acceleration=0.2,
        mouse_jitter=0.8,
        mouse_curve_factor=0.2,
        click_precision=0.92,
        double_click_speed=120,
        action_interval_range=(80, 300),
        typing_speed_range=(25, 60),
        typing_variance=0.15,
        hesitation_probability=0.05,
        scroll_pattern="smooth",
        focus_changes=False,
        micro_corrections=False,
        fatigue_simulation=False,
        randomization_level=0.2,
        human_noise_injection=0.05
    ),
    
    "puppeteer_stealth": BotProfile(
        name="Puppeteer-Extra Stealth",
        bot_type=BotType.PUPPETEER,
        description="Puppeteer with stealth plugin",
        mouse_speed_range=(0.8, 2.5),
        mouse_acceleration=0.5,
        mouse_jitter=1.5,
        mouse_curve_factor=0.5,
        click_precision=0.85,
        double_click_speed=180,
        action_interval_range=(150, 600),
        typing_speed_range=(40, 100),
        typing_variance=0.3,
        hesitation_probability=0.15,
        scroll_pattern="variable",
        focus_changes=True,
        micro_corrections=True,
        fatigue_simulation=False,
        randomization_level=0.5,
        human_noise_injection=0.2
    ),
    
    # Level 4: Macro Tools
    "autohotkey": BotProfile(
        name="AutoHotkey Script",
        bot_type=BotType.MACRO,
        description="Desktop automation macro",
        mouse_speed_range=(3.0, 6.0),  # Often faster than humans
        mouse_acceleration=0.0,
        mouse_jitter=0.2,
        mouse_curve_factor=0.0,  # Teleport-like movements
        click_precision=1.0,
        double_click_speed=80,
        action_interval_range=(50, 150),
        typing_speed_range=(15, 30),  # Very fast typing
        typing_variance=0.05,
        hesitation_probability=0.0,
        scroll_pattern="instant",
        focus_changes=False,
        micro_corrections=False,
        fatigue_simulation=False,
        randomization_level=0.05,
        human_noise_injection=0.0
    ),
    
    # Level 5: Credential Stuffing Tools
    "credential_stuffer": BotProfile(
        name="Credential Stuffing Bot",
        bot_type=BotType.CREDENTIAL_STUFFER,
        description="Automated login attack tool",
        mouse_speed_range=(4.0, 4.5),
        mouse_acceleration=0.0,
        mouse_jitter=0.0,
        mouse_curve_factor=0.0,
        click_precision=1.0,
        double_click_speed=50,
        action_interval_range=(20, 50),  # Very fast
        typing_speed_range=(5, 15),  # Paste-like speed
        typing_variance=0.0,
        hesitation_probability=0.0,
        scroll_pattern="instant",
        focus_changes=False,
        micro_corrections=False,
        fatigue_simulation=False,
        randomization_level=0.0,
        human_noise_injection=0.0
    ),
    
    # Level 6: Click Farms (Human-Assisted)
    "click_farm_basic": BotProfile(
        name="Click Farm Worker",
        bot_type=BotType.CLICK_FARM,
        description="Human workers with scripts - repetitive patterns",
        mouse_speed_range=(0.5, 2.0),
        mouse_acceleration=0.7,
        mouse_jitter=2.0,
        mouse_curve_factor=0.7,
        click_precision=0.8,
        double_click_speed=200,
        action_interval_range=(200, 800),
        typing_speed_range=(80, 200),  # Slow, tired workers
        typing_variance=0.4,
        hesitation_probability=0.2,
        scroll_pattern="variable",
        focus_changes=True,
        micro_corrections=True,
        fatigue_simulation=True,  # Gets slower over time
        randomization_level=0.6,
        human_noise_injection=0.5
    ),
    
    # Level 7: Mobile Emulators
    "mobile_emulator": BotProfile(
        name="Android Emulator Bot",
        bot_type=BotType.MOBILE_EMU,
        description="Appium/ADB automation on emulators",
        mouse_speed_range=(1.0, 2.0),  # Touch is different
        mouse_acceleration=0.4,
        mouse_jitter=3.0,  # Touch is imprecise
        mouse_curve_factor=0.3,
        click_precision=0.75,  # Touch is less accurate
        double_click_speed=250,
        action_interval_range=(100, 400),
        typing_speed_range=(50, 150),  # On-screen keyboard
        typing_variance=0.3,
        hesitation_probability=0.1,
        scroll_pattern="smooth",
        focus_changes=False,
        micro_corrections=False,
        fatigue_simulation=False,
        randomization_level=0.3,
        human_noise_injection=0.1
    ),
    
    # Level 8: Evasive Bots
    "evasive_bot": BotProfile(
        name="Evasive Bot",
        bot_type=BotType.EVASIVE,
        description="Bot designed to evade basic detection",
        mouse_speed_range=(0.6, 2.2),
        mouse_acceleration=0.6,
        mouse_jitter=2.5,
        mouse_curve_factor=0.6,
        click_precision=0.82,
        double_click_speed=220,
        action_interval_range=(180, 700),
        typing_speed_range=(50, 120),
        typing_variance=0.35,
        hesitation_probability=0.2,
        scroll_pattern="variable",
        focus_changes=True,
        micro_corrections=True,
        fatigue_simulation=True,
        randomization_level=0.7,
        human_noise_injection=0.4
    ),
    
    # Level 9: Advanced AI-Powered Evasion
    "advanced_evasive": BotProfile(
        name="AI-Powered Evasive Bot",
        bot_type=BotType.ADVANCED_EVASIVE,
        description="State-of-the-art GAN-based human mimicry",
        mouse_speed_range=(0.4, 2.0),
        mouse_acceleration=0.8,
        mouse_jitter=3.0,
        mouse_curve_factor=0.85,
        click_precision=0.78,
        double_click_speed=260,
        action_interval_range=(200, 1200),
        typing_speed_range=(60, 180),
        typing_variance=0.45,
        hesitation_probability=0.3,
        scroll_pattern="variable",
        focus_changes=True,
        micro_corrections=True,
        fatigue_simulation=True,
        randomization_level=0.9,
        human_noise_injection=0.7
    )
}


class AttackSimulator:
    """
    Generates realistic bot behavioral data based on attack tool profiles.
    Used for training robust detection models.
    """
    
    def __init__(self, human_templates: Optional[List[Dict]] = None):
        """
        Args:
            human_templates: Real human sessions to use as reference for evasive bots
        """
        self.human_templates = human_templates or []
        self.profiles = BOT_PROFILES
        
    def generate_session(
        self,
        profile_name: str,
        duration_ms: int = 30000,
        num_clicks: int = 10,
        num_keystrokes: int = 50
    ) -> Dict:
        """Generate a bot session using the specified profile"""
        
        profile = self.profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")
            
        session = {
            "mouse_movements": [],
            "clicks": [],
            "keystrokes": [],
            "scrolls": [],
            "metadata": {
                "bot_type": profile.bot_type.value,
                "profile_name": profile_name,
                "description": profile.description,
                "duration_ms": duration_ms
            }
        }
        
        # Generate mouse movements
        session["mouse_movements"] = self._generate_mouse_movements(
            profile, duration_ms
        )
        
        # Generate clicks
        session["clicks"] = self._generate_clicks(
            profile, session["mouse_movements"], num_clicks
        )
        
        # Generate keystrokes
        session["keystrokes"] = self._generate_keystrokes(
            profile, num_keystrokes
        )
        
        # Generate scrolls
        session["scrolls"] = self._generate_scrolls(profile, duration_ms)
        
        return session
    
    def _generate_mouse_movements(
        self,
        profile: BotProfile,
        duration_ms: int
    ) -> List[Dict]:
        """Generate mouse movement trajectory"""
        
        movements = []
        current_time = 0
        current_x, current_y = 500, 400  # Start position
        
        # Screen boundaries
        max_x, max_y = 1920, 1080
        
        while current_time < duration_ms:
            # Determine next target
            target_x = random.randint(50, max_x - 50)
            target_y = random.randint(50, max_y - 50)
            
            # Calculate distance
            dx = target_x - current_x
            dy = target_y - current_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate movement speed
            speed = random.uniform(*profile.mouse_speed_range)
            move_duration = distance / speed if speed > 0 else 100
            
            # Generate intermediate points
            num_points = max(2, int(move_duration / 16))  # ~60fps
            
            for i in range(num_points):
                t = i / (num_points - 1) if num_points > 1 else 1
                
                # Apply curve factor (Bezier-like)
                if profile.mouse_curve_factor > 0:
                    # Add curve via control point
                    ctrl_x = current_x + dx * 0.5 + random.gauss(0, distance * 0.2 * profile.mouse_curve_factor)
                    ctrl_y = current_y + dy * 0.5 + random.gauss(0, distance * 0.2 * profile.mouse_curve_factor)
                    
                    # Quadratic bezier
                    point_x = (1-t)**2 * current_x + 2*(1-t)*t * ctrl_x + t**2 * target_x
                    point_y = (1-t)**2 * current_y + 2*(1-t)*t * ctrl_y + t**2 * target_y
                else:
                    # Straight line
                    point_x = current_x + dx * t
                    point_y = current_y + dy * t
                
                # Apply acceleration
                if profile.mouse_acceleration > 0:
                    # Ease in-out
                    ease_t = t * t * (3 - 2 * t) * profile.mouse_acceleration + t * (1 - profile.mouse_acceleration)
                    time_offset = move_duration * ease_t / num_points
                else:
                    time_offset = move_duration / num_points
                
                # Apply jitter
                if profile.mouse_jitter > 0:
                    point_x += random.gauss(0, profile.mouse_jitter)
                    point_y += random.gauss(0, profile.mouse_jitter)
                
                # Add human noise
                if profile.human_noise_injection > 0 and random.random() < profile.human_noise_injection * 0.1:
                    # Micro-hesitation
                    time_offset += random.randint(10, 50)
                
                movements.append({
                    "x": max(0, min(max_x, point_x)),
                    "y": max(0, min(max_y, point_y)),
                    "timestamp": current_time + time_offset * i,
                    "velocity": speed
                })
            
            current_time += move_duration
            current_x, current_y = target_x, target_y
            
            # Micro-corrections
            if profile.micro_corrections and random.random() < 0.3:
                for _ in range(random.randint(1, 3)):
                    movements.append({
                        "x": current_x + random.gauss(0, 3),
                        "y": current_y + random.gauss(0, 3),
                        "timestamp": current_time + random.randint(10, 50),
                        "velocity": 0.5
                    })
                current_time += 50
            
            # Pause between movements
            pause = random.randint(*profile.action_interval_range)
            if profile.randomization_level > 0:
                pause = int(pause * (1 + random.gauss(0, profile.randomization_level * 0.3)))
            current_time += max(10, pause)
        
        return movements
    
    def _generate_clicks(
        self,
        profile: BotProfile,
        movements: List[Dict],
        num_clicks: int
    ) -> List[Dict]:
        """Generate click events"""
        
        clicks = []
        if not movements:
            return clicks
            
        # Pick positions from movement trajectory
        movement_times = [m["timestamp"] for m in movements]
        duration = max(movement_times) if movement_times else 30000
        
        for i in range(num_clicks):
            click_time = (duration / num_clicks) * i + random.randint(100, 500)
            
            # Find nearest mouse position
            nearest_idx = min(
                range(len(movements)),
                key=lambda j: abs(movements[j]["timestamp"] - click_time)
            )
            
            pos = movements[nearest_idx]
            
            # Apply precision offset
            offset = (1 - profile.click_precision) * 20
            click_x = pos["x"] + random.gauss(0, offset)
            click_y = pos["y"] + random.gauss(0, offset)
            
            clicks.append({
                "x": click_x,
                "y": click_y,
                "timestamp": click_time,
                "button": "left",
                "type": "click"
            })
            
            # Sometimes double-click
            if random.random() < 0.1:
                clicks.append({
                    "x": click_x + random.gauss(0, 2),
                    "y": click_y + random.gauss(0, 2),
                    "timestamp": click_time + profile.double_click_speed,
                    "button": "left",
                    "type": "dblclick"
                })
        
        return sorted(clicks, key=lambda c: c["timestamp"])
    
    def _generate_keystrokes(
        self,
        profile: BotProfile,
        num_keystrokes: int
    ) -> List[Dict]:
        """Generate keystroke events"""
        
        keystrokes = []
        current_time = random.randint(500, 2000)  # Start after some mouse activity
        
        # Simulate typing a password-like string
        sample_text = "P@ssw0rd123!Example"
        
        for i in range(num_keystrokes):
            char = sample_text[i % len(sample_text)]
            
            # Key timing
            base_interval = random.randint(*profile.typing_speed_range)
            
            # Apply variance
            if profile.typing_variance > 0:
                variance = int(base_interval * profile.typing_variance)
                base_interval += random.randint(-variance, variance)
            
            # Hesitation before action
            if profile.hesitation_probability > 0 and random.random() < profile.hesitation_probability:
                current_time += random.randint(200, 800)
            
            # Key down
            keystrokes.append({
                "key": char,
                "type": "keydown",
                "timestamp": current_time
            })
            
            # Hold time (human: 50-150ms, bot: very short)
            if profile.bot_type in [BotType.NAIVE, BotType.CREDENTIAL_STUFFER]:
                hold_time = random.randint(5, 20)
            else:
                hold_time = random.randint(30, 120)
            
            # Key up
            keystrokes.append({
                "key": char,
                "type": "keyup",
                "timestamp": current_time + hold_time
            })
            
            current_time += base_interval
            
            # Fatigue simulation
            if profile.fatigue_simulation and i > num_keystrokes * 0.7:
                current_time += random.randint(0, 100)  # Slow down
        
        return keystrokes
    
    def _generate_scrolls(
        self,
        profile: BotProfile,
        duration_ms: int
    ) -> List[Dict]:
        """Generate scroll events"""
        
        scrolls = []
        current_time = random.randint(1000, 3000)
        
        while current_time < duration_ms:
            scroll_amount = random.choice([-100, -200, -300, 100, 200, 300])
            
            if profile.scroll_pattern == "instant":
                # Single large scroll
                scrolls.append({
                    "deltaY": scroll_amount,
                    "timestamp": current_time
                })
                current_time += random.randint(2000, 5000)
                
            elif profile.scroll_pattern == "smooth":
                # Many small scrolls
                num_steps = abs(scroll_amount) // 20
                for j in range(num_steps):
                    scrolls.append({
                        "deltaY": 20 if scroll_amount > 0 else -20,
                        "timestamp": current_time + j * 16  # 60fps
                    })
                current_time += num_steps * 16 + random.randint(1000, 3000)
                
            elif profile.scroll_pattern == "stepped":
                # Medium chunks
                num_steps = abs(scroll_amount) // 50
                for j in range(num_steps):
                    scrolls.append({
                        "deltaY": 50 if scroll_amount > 0 else -50,
                        "timestamp": current_time + j * 100
                    })
                current_time += num_steps * 100 + random.randint(1000, 3000)
                
            else:  # variable
                # Mix of patterns
                pattern = random.choice(["instant", "smooth", "stepped"])
                if pattern == "instant":
                    scrolls.append({
                        "deltaY": scroll_amount,
                        "timestamp": current_time
                    })
                else:
                    num_steps = random.randint(2, 6)
                    step_size = scroll_amount // num_steps
                    for j in range(num_steps):
                        scrolls.append({
                            "deltaY": step_size,
                            "timestamp": current_time + j * random.randint(30, 100)
                        })
                current_time += random.randint(1500, 4000)
        
        return scrolls
    
    def generate_dataset(
        self,
        samples_per_profile: int = 50,
        profiles: Optional[List[str]] = None
    ) -> Tuple[List[Dict], List[int]]:
        """
        Generate a complete training dataset with all bot types.
        
        Returns:
            sessions: List of session dictionaries
            labels: List of labels (1 = bot for all)
        """
        
        if profiles is None:
            profiles = list(self.profiles.keys())
        
        sessions = []
        labels = []
        
        for profile_name in profiles:
            print(f"  Generating {samples_per_profile} samples for: {profile_name}")
            
            for i in range(samples_per_profile):
                # Vary session parameters
                duration = random.randint(15000, 60000)
                num_clicks = random.randint(5, 25)
                num_keystrokes = random.randint(20, 100)
                
                session = self.generate_session(
                    profile_name,
                    duration_ms=duration,
                    num_clicks=num_clicks,
                    num_keystrokes=num_keystrokes
                )
                
                sessions.append(session)
                labels.append(1)  # All are bots
        
        return sessions, labels
    
    def generate_adversarial_samples(
        self,
        human_sessions: List[Dict],
        num_samples: int = 100
    ) -> Tuple[List[Dict], List[int]]:
        """
        Generate adversarial bot samples that try to mimic human patterns.
        These are the hardest to detect.
        """
        
        sessions = []
        labels = []
        
        if not human_sessions:
            print("Warning: No human templates provided for adversarial generation")
            return sessions, labels
        
        print(f"  Generating {num_samples} adversarial samples from {len(human_sessions)} human templates")
        
        for i in range(num_samples):
            # Pick a random human session as template
            template = random.choice(human_sessions)
            
            # Create adversarial copy with subtle bot artifacts
            adv_session = self._create_adversarial_copy(template)
            
            sessions.append(adv_session)
            labels.append(1)  # Still a bot
        
        return sessions, labels
    
    def _create_adversarial_copy(self, human_session: Dict) -> Dict:
        """Create an adversarial bot copy of a human session"""
        
        session = {
            "mouse_movements": [],
            "clicks": [],
            "keystrokes": [],
            "scrolls": [],
            "metadata": {
                "bot_type": "adversarial",
                "description": "GAN-like human mimicry with subtle artifacts"
            }
        }
        
        # Copy mouse movements with subtle artifacts
        if "mouse_movements" in human_session:
            for m in human_session.get("mouse_movements", []):
                new_m = m.copy()
                # Slight reduction in natural variance (bots are too smooth)
                if random.random() < 0.2:
                    new_m["x"] = round(new_m.get("x", 0), 1)  # Over-precise coordinates
                    new_m["y"] = round(new_m.get("y", 0), 1)
                session["mouse_movements"].append(new_m)
        
        # Copy clicks with timing artifacts
        for c in human_session.get("clicks", []):
            new_c = c.copy()
            # Bots often have suspiciously consistent click-release timing
            if random.random() < 0.3:
                new_c["timestamp"] = round(new_c.get("timestamp", 0) / 10) * 10
            session["clicks"].append(new_c)
        
        # Copy keystrokes with subtle patterns
        for k in human_session.get("keystrokes", []):
            new_k = k.copy()
            # Bots may have too-consistent inter-key timing
            if random.random() < 0.25:
                new_k["timestamp"] = round(new_k.get("timestamp", 0) / 5) * 5
            session["keystrokes"].append(new_k)
        
        # Copy scrolls
        session["scrolls"] = human_session.get("scrolls", []).copy()
        
        return session


def generate_attack_dataset(
    output_path: Optional[str] = None,
    samples_per_profile: int = 50,
    include_adversarial: bool = True,
    human_templates: Optional[List[Dict]] = None
) -> Tuple[List[Dict], List[int]]:
    """
    Convenience function to generate a complete attack dataset.
    
    Args:
        output_path: Optional path to save dataset
        samples_per_profile: Number of samples per bot type
        include_adversarial: Whether to include adversarial samples
        human_templates: Human sessions for adversarial generation
        
    Returns:
        (sessions, labels) tuple
    """
    
    print("=" * 60)
    print("GENERATING ATTACK DATASET")
    print("=" * 60)
    
    simulator = AttackSimulator(human_templates)
    
    # Generate all bot types
    all_sessions = []
    all_labels = []
    
    print("\n[1/2] Generating bot samples by attack type...")
    sessions, labels = simulator.generate_dataset(samples_per_profile)
    all_sessions.extend(sessions)
    all_labels.extend(labels)
    
    # Generate adversarial samples
    if include_adversarial and human_templates:
        print("\n[2/2] Generating adversarial samples...")
        adv_sessions, adv_labels = simulator.generate_adversarial_samples(
            human_templates,
            num_samples=samples_per_profile * 2  # More adversarial samples
        )
        all_sessions.extend(adv_sessions)
        all_labels.extend(adv_labels)
    
    print(f"\n{'=' * 60}")
    print(f"ATTACK DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total bot samples: {len(all_sessions)}")
    print(f"Attack types: {len(BOT_PROFILES)}")
    if include_adversarial and human_templates:
        print(f"Adversarial samples: {samples_per_profile * 2}")
    
    return all_sessions, all_labels


if __name__ == "__main__":
    # Test generation
    print("Testing Attack Simulator...")
    
    simulator = AttackSimulator()
    
    # Generate one sample from each profile
    for name, profile in BOT_PROFILES.items():
        session = simulator.generate_session(name, duration_ms=5000)
        print(f"\n{name}:")
        print(f"  Mouse points: {len(session['mouse_movements'])}")
        print(f"  Clicks: {len(session['clicks'])}")
        print(f"  Keystrokes: {len(session['keystrokes'])}")
        print(f"  Scrolls: {len(session['scrolls'])}")
