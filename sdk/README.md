# Turing Defense SDK

**Invisible bot detection for any website.** No CAPTCHAs, no puzzles—just behavioral analysis.

## Quick Start

### 1. Include the SDK

Add this script tag to your HTML (before `</body>`):

```html
<script src="https://your-domain.com/sdk/turing-defense.js" 
        data-endpoint="https://your-api.com"
        data-site-key="your-site-key"></script>
```

That's it! The SDK will automatically:
- Start collecting behavioral data (mouse, keyboard, clicks, scroll)
- Send data to your backend for analysis
- Provide a bot score you can use

### 2. Protect Your Forms

Add a hidden field and check the score before submission:

```html
<form id="loginForm">
    <input type="email" name="email" required>
    <input type="password" name="password" required>
    
    <!-- Hidden token field -->
    <input type="hidden" name="turing_token" id="turingToken">
    
    <button type="submit">Log In</button>
</form>

<script>
document.getElementById('loginForm').addEventListener('submit', function(e) {
    // Get the bot score
    const score = TuringDefense.getScore();
    const token = TuringDefense.getToken();
    
    // Block if likely a bot (score > 50)
    if (score > 50) {
        e.preventDefault();
        alert('Please try again');
        return;
    }
    
    // Include token for server-side verification
    document.getElementById('turingToken').value = token;
});
</script>
```

### 3. Verify on Your Server

```python
# Python/Flask example
import requests

@app.route('/login', methods=['POST'])
def login():
    token = request.form.get('turing_token')
    
    # Verify with Turing Defense API
    response = requests.post('https://your-api.com/api/verify-token', 
        json={'token': token}
    )
    
    if not response.json().get('valid'):
        return 'Bot detected', 403
    
    # Continue with normal login...
```

---

## API Reference

### Configuration Options

| Attribute | Description | Default |
|-----------|-------------|---------|
| `data-endpoint` | Your Turing Defense API URL | Required |
| `data-site-key` | Your site identifier | `''` |
| `data-auto-start` | Start collecting immediately | `true` |
| `data-debug` | Enable console logging | `false` |

### JavaScript API

```javascript
// Get current bot score (0-100, higher = more bot-like)
const score = TuringDefense.getScore();

// Check if user is likely a bot
const isBot = TuringDefense.isBot(); // true if score > 50

// Get confidence level (0-100)
const confidence = TuringDefense.getConfidence();

// Get verification token for form submission
const token = TuringDefense.getToken();

// Get full analysis result
const result = TuringDefense.getResult();
// { bot_score, is_bot, confidence, triggers, token, ... }

// Listen for analysis results
TuringDefense.onResult(function(result) {
    console.log('Bot score:', result.bot_score);
});

// Get session statistics
const stats = TuringDefense.getStats();
// { mousePoints, keystrokes, clicks, scrollEvents, ... }

// Force analysis now
await TuringDefense.analyze();

// Reset session
TuringDefense.reset();

// Manual initialization (if auto-start is disabled)
TuringDefense.init({
    endpoint: 'https://your-api.com',
    siteKey: 'your-site-key',
    debug: true
});
```

---

## What Gets Collected

The SDK collects **behavioral timing data only**, not content:

| Signal | What's Collected | Privacy |
|--------|-----------------|---------|
| **Mouse** | Movement coordinates, velocity, timing | ✅ No PII |
| **Keyboard** | Key timing, intervals (NOT key values) | ✅ No content |
| **Clicks** | Position, timing, target element type | ✅ No content |
| **Scroll** | Direction, velocity, patterns | ✅ No content |

### What Makes Bots Detectable

| Human Behavior | Bot Behavior |
|----------------|--------------|
| Variable mouse speed | Constant velocity |
| Curved paths | Straight lines |
| Hesitations & corrections | Perfect accuracy |
| Irregular typing rhythm | Constant intervals |
| Natural scroll momentum | Fixed scroll amounts |

---

## Server Integration

### Backend Endpoints

Your Turing Defense backend provides these endpoints:

#### `POST /api/analyze`
Analyze behavioral data and get a bot score.

```bash
curl -X POST https://your-api.com/api/analyze \
  -H "Content-Type: application/json" \
  -H "X-Site-Key: your-site-key" \
  -d '{"session_id": "abc123", "data": {...}}'
```

Response:
```json
{
    "status": "analyzed",
    "bot_score": 23.5,
    "is_bot": false,
    "confidence": 87.2,
    "token": "td_1703789012_a1b2c3d4...",
    "triggers": [],
    "component_scores": {
        "mouse": 0.15,
        "keyboard": 0.22,
        "click": 0.18,
        "scroll": 0.25
    }
}
```

#### `POST /api/verify-token`
Verify a token from form submission.

```bash
curl -X POST https://your-api.com/api/verify-token \
  -H "Content-Type: application/json" \
  -d '{"token": "td_1703789012_a1b2c3d4..."}'
```

Response:
```json
{
    "valid": true,
    "token": "td_1703789012_a1b2c3d4..."
}
```

---

## Self-Hosting

### Deploy with Docker

```bash
git clone https://github.com/yourusername/turing-defense.git
cd turing-defense
docker-compose up -d
```

### Environment Variables

```bash
TURING_SECRET_KEY=your-secret-key-for-tokens
```

### Serve the SDK

The SDK is served from `/sdk/turing-defense.js` on your backend.

---

## Use Cases

- **Login Forms** - Block credential stuffing bots
- **Registration** - Prevent fake account creation
- **E-commerce** - Stop scalper bots
- **Contact Forms** - Filter spam submissions
- **Voting/Polling** - Ensure one-human-one-vote
- **Content** - Detect scraping bots

---

## Browser Support

- Chrome 60+
- Firefox 55+
- Safari 11+
- Edge 79+

---

## License

MIT License - Built by **Ashmith Maddala**
