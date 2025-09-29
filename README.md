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

## Quick Start

### 1. Clone
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

## 2. Create & activate virtualenv
# macOS/Linux
python3.11 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

## 3. Install dependencies
pip install -r requirements.txt

## 4. Configure environment
cp .env.example .env
# edit .env and set your API_KEY

## 5. Run
python email_agent_llama.py

## Example Usage

### Database
orders = [
    "Alex, wine A, 2025-09-29",
    "Alex, wine B, 2025-07-05",
    "David, wine B, 2025-09-29",    
    "David, wine A, 2025-05-29",
]
inventory = ["wine A, 5", "wine B, 3"]

Input email: Hi, I'm Alex, I want to order 2 wines I ordered last time.
Output response: Alex's last order was for wine A on 2025-09-29. The current stock comparison for the requested wine type shows that there are 3 units of wine B available, which is enough to fulfill the requested quantity of 2. Given that the available stock exceeds the requested amount, I suggest proceeding with invoicing the customer for 2 units of wine B.

Input email: Hello, this is David. I’d like to order the same two wines I purchased last time.
Output response: David's last order was for wine B on 2025-09-29, and we currently have 3 units of wine B in stock. Since David is requesting 2 units of wine, which is less than the available stock, we have enough to fulfill the order. Considering the available stock and the request, I suggest proceeding with the invoice for 2 units of wine B to complete David's purchase.

Input email: Good afternoon, my name is David. I’d like to order the same five bottles of wine that I got during my last order, as I’d like to enjoy them again.
Output response: David's last order was for wine B on 2025-09-29, with a current stock quantity of 3. Given the current comparison, David has requested 5 items but only 3 are available, indicating that there is not enough stock to fulfill the request. Since there is a shortage of 2 items, I suggest issuing a backorder for the remaining 2 items to ensure David receives his full order as soon as possible.
