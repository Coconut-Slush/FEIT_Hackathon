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
