# Email Agent (LLaMA/Qwen compatible)

Small, hackathon-ready agent that:
1) Parses a free-form email,
2) Extracts order intent (name, quantity, type, time),
3) Looks up matching info from simple in-memory "orders" & "inventory" stores,
4) Returns a suggested reply.

## Project Status
- Minimal working prototype for demo.
- Python 3.11 tested.
- Uses a free LLM API (configurable) via `requests`.

## Requirements
- Python **3.11.0**
- `requests==2.32.5`
- `python-dotenv==1.1.1`

Install via:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Clone
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

## 2. Create & activate virtualenv
# macOS/Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

# Windows (PowerShell)
```bash
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Configure environment
```bash
cp .env.example .env
# edit .env and set your API_KEY
```

## 5. Run
```bash
python email_agent_llama.py
```

## Example Usage
```python
python test_email.agent.py
```
example output in example_output.txt
