# Turing Defense

**Behavioral Biometrics Bot Detection System**

A production-grade machine learning platform that detects automated bots by analyzing human behavioral patterns in real-time. Uses neural networks trained on research datasets with adversarial robustness techniques.

![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![React 19](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)

---

## Overview

Turing Defense implements behavioral biometrics analysis to distinguish humans from bots without CAPTCHAs or user friction. The system analyzes mouse dynamics, keystroke patterns, and interaction timing to detect automation.

### Key Capabilities

- **Real-time Detection** — Sub-second analysis via WebSocket streaming
- **Neural Network Model** — 56-feature deep learning classifier with 91.8% accuracy
- **Adversarial Robustness** — Resistant to evasion attacks (FGSM, PGD)
- **11 Attack Profiles** — Trained against Selenium, Puppeteer, credential stuffers, and more
- **Zero False Positives** — 0% FPR ensures legitimate users aren't blocked

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Turing Defense                              │
├──────────────────────────┬──────────────────────────────────────┤
│       React Client       │           Flask Backend              │
│                          │                                      │
│  ┌────────────────────┐  │  ┌────────────────────────────────┐ │
│  │ Behavioral Capture │──┼─►│ Feature Extraction (56 dims)   │ │
│  │ • Mouse movements  │  │  │ • Kinematics (velocity, accel) │ │
│  │ • Keystrokes       │  │  │ • Geometry (curvature, paths)  │ │
│  │ • Click patterns   │  │  │ • Temporal (timing entropy)    │ │
│  │ • Scroll events    │  │  │ • Micro-patterns (corrections) │ │
│  └────────────────────┘  │  └──────────────┬─────────────────┘ │
│                          │                 ▼                    │
│  ┌────────────────────┐  │  ┌────────────────────────────────┐ │
│  │ Detection UI       │◄─┼──│ Neural Network Classifier      │ │
│  │ • Confidence score │  │  │ • 3 hidden layers (128→64→32)  │ │
│  │ • Feature breakdown│  │  │ • BatchNorm + Dropout          │ │
│  │ • Real-time graphs │  │  │ • Adversarially trained        │ │
│  └────────────────────┘  │  └────────────────────────────────┘ │
│        :5173             │              :5000                   │
└──────────────────────────┴──────────────────────────────────────┘
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 91.8% |
| Precision | 100.0% |
| Recall | 83.3% |
| F1 Score | 0.909 |
| False Positive Rate | 0.0% |

### Adversarial Robustness

Model maintains accuracy under gradient-based attacks:

| Attack Strength (ε) | Accuracy |
|---------------------|----------|
| Clean (ε=0) | 91.8% |
| ε=0.05 | 91.8% |
| ε=0.10 | 91.8% |
| ε=0.20 | 91.8% |

---

## Training Data

### Research Datasets
- **CMU Keystroke Dynamics** — 51 subjects, 2,550 typing sessions
- **Balabit Mouse Dynamics** — 10 users, 65 sessions, 30K+ mouse events each

### Simulated Attack Types
| Category | Profiles |
|----------|----------|
| Browser Automation | Selenium, Puppeteer, Playwright |
| Stealth Variants | undetected-chromedriver, puppeteer-stealth |
| Desktop Macros | AutoHotkey, AutoIt |
| Attack Tools | Credential stuffers, click farms |
| Mobile | Android emulator automation |
| Evasive | AI-powered human mimicry |

---

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- PyTorch

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup

```bash
cd client
npm install
npm run dev
```

### Docker Deployment

```bash
docker-compose up --build
```

Access the application at `http://localhost:5173`

---

## Project Structure

```
turing-defense/
├── backend/
│   ├── app.py                    # Flask server + WebSocket
│   ├── ml_neural_network.py      # Neural network model
│   ├── ml_model_advanced.py      # Heuristic fallback
│   ├── attack_simulators.py      # Bot behavior simulation
│   ├── adversarial_training.py   # FGSM/PGD training
│   ├── production_collector.py   # Data collection pipeline
│   ├── dataset_loaders.py        # CMU/Balabit loaders
│   ├── train_robust_model.py     # Training pipeline
│   ├── models/                   # Trained model weights
│   └── datasets/                 # Training data
├── client/
│   ├── src/
│   │   ├── App.jsx               # Main application
│   │   └── App.css               # Styles
│   └── package.json
├── sdk/
│   ├── turing-defense.js         # Embeddable SDK
│   ├── example.html              # Integration demo
│   └── README.md                 # SDK documentation
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## Embeddable SDK

Turing Defense can be integrated into **any website** with a simple script tag.

### Quick Integration

```html
<script src="https://your-domain.com/sdk/turing-defense.js" 
        data-endpoint="https://your-api.com"
        data-site-key="your-site-key"></script>
```

### Usage

```javascript
// Get bot score (0-100)
const score = TuringDefense.getScore();

// Check if likely a bot
if (TuringDefense.isBot()) {
    blockSubmission();
}

// Get token for server-side verification
const token = TuringDefense.getToken();
```

### Demo

Access the SDK demo at `http://localhost:5000/sdk/example`

See [sdk/README.md](sdk/README.md) for full documentation.

---

## API Reference

### WebSocket Events (Real-time Demo)

**Client → Server**
```javascript
socket.emit('behavioral_data', {
  mouse_data: [...],      // {x, y, timestamp, velocity}
  keyboard_data: [...],   // {key, type, timestamp}
  click_data: [...],      // {x, y, timestamp, button}
  scroll_data: [...]      // {deltaY, timestamp}
});
```

**Server → Client**
```javascript
socket.on('analysis_result', {
  is_bot: false,
  confidence: 0.12,
  verdict: "HUMAN",
  features: {...},
  triggers: []
});
```

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Analyze behavioral data |
| GET | `/api/metrics` | Get model metrics |
| GET | `/api/health` | Health check |

---

## Technical Details

### Feature Engineering

The model extracts 56 behavioral features across 8 categories:

1. **Mouse Kinematics** (12) — Velocity, acceleration, jerk statistics
2. **Mouse Geometry** (10) — Path straightness, curvature, Bézier fit
3. **Mouse Temporal** (8) — Timing entropy, pauses, bursts
4. **Micro-patterns** (6) — Corrections, hesitations, tremor
5. **Keyboard** (8) — Inter-key timing, digraph latency, hold duration
6. **Click** (4) — Click timing, precision, double-click patterns
7. **Scroll** (4) — Scroll velocity, direction changes
8. **Session** (4) — Duration, activity density, consistency

### Neural Network Architecture

```
Input (56) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
          → Linear(32)  → BatchNorm → ReLU → Dropout(0.3)
          → Linear(1)   → Sigmoid
```

### Adversarial Training

- **FGSM** — Fast Gradient Sign Method for efficient perturbations
- **PGD** — Projected Gradient Descent for stronger attacks
- **Curriculum Learning** — Progressive difficulty increase
- **Data Augmentation** — Mixup, noise injection, feature dropout

---

## Testing

```bash
cd backend
pytest tests/ -v
```

---

## References

1. Killourhy, K. & Maxion, R. (2009). *Comparing Anomaly-Detection Algorithms for Keystroke Dynamics*. IEEE DSN.
2. Antal, M. et al. (2019). *Intrusion Detection Using Mouse Dynamics*. IET Biometrics.
3. Madry, A. et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR.

---

## License

This project is proprietary software. See the [LICENSE](LICENSE) file for details.

**© 2025 Ashmith Maddala. All rights reserved.**

