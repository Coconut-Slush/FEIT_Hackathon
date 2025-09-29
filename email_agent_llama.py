import os, json, re, requests
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # this reads .env into environment variables

def parse_json_from_text(s: str) -> dict:
    import re, json
    if not s:
        raise ValueError("Empty LLM output")
    t = s.strip()
    # Strip markdown fences
    t = re.sub(r'^```(?:json)?', '', t.strip(), flags=re.IGNORECASE)
    t = re.sub(r'```$', '', t.strip())
    # Extract first {...}
    start, end = t.find('{'), t.rfind('}')
    if start != -1 and end != -1:
        t = t[start:end+1]
    return json.loads(t)

# ----------------------- Data classes -----------------------
@dataclass
class Extracted:
    name: Optional[str]
    number: Optional[int]
    type: Optional[str]
    purchase_time: Optional[str]

@dataclass
class Retrieval:
    last_sku: Optional[str]
    last_date: Optional[str]
    stock_sku: Optional[str]
    stock_qty: Optional[int]

# ----------------------- Llama API client -----------------------
def call_llama(messages, model="llama-3.3-70b-versatile", temperature=0.0):
    """Call Groq (OpenAI-compatible) with a Llama-3 model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ----------------------- Prompts -----------------------
SYSTEM_EXTRACT = """You are an information extraction model.
Return STRICT JSON with keys: name, number, type, purchase_time.
- name: string or null
- number: integer or null
- type: string or null (be as specific as possible, e.g., "wine B").
  If the email says "same as last time" or equivalent, set type = null.
- purchase_time: string or null (normalize "last time"/"previous order" into "last time").
Output ONLY JSON, no commentary.
"""
def USER_EXTRACT(text): return f"Text: {text}\nReturn JSON now."

SYSTEM_RETRIEVE = """You are a retrieval selection assistant.
Return a SINGLE JSON object with keys:
- last_sku (string or null)
- last_date (string or null, YYYY-MM-DD)
- stock_sku (string or null)
- stock_qty (integer or null)

RULES:
- Output ONLY the JSON object.
- Do NOT include explanations, steps, reasoning, or code fences.
- Do NOT include any text outside the JSON.
"""
def USER_RETRIEVE(extracted, orders, inventory):
    # print("USER_RETRIEVE inputs:", extracted, orders, inventory)
    return json.dumps({"extracted":extracted,"orders":orders,"inventory":inventory}, ensure_ascii=False)

SYSTEM_OUTPUT = """You are a concise business assistant.
Given extracted fields, chosen retrieval, and stock comparison, write ONE paragraph.
Mention last order and stock if known. End with a clear suggestion (e.g., invoice or backorder).
"""
def USER_OUTPUT(extracted, retrieval, comparison):
    return json.dumps({"extracted":extracted,"retrieval":retrieval,"comparison":comparison}, ensure_ascii=False)

# ----------------------- Pipeline -----------------------
def extract_llama(text: str) -> Extracted:
    content = call_llama([{"role":"system","content":SYSTEM_EXTRACT},
                          {"role":"user","content":USER_EXTRACT(text)}])
    # print("Extraction LLM output:", content)
    data = parse_json_from_text(content)
    return Extracted(**data)

def retrieve_llama(extracted: Extracted, orders: List[Dict[str,str]], inventory: Dict[str,int]) -> Retrieval:
    # If type is missing but purchase_time = last time → resolve directly
    if not extracted.type and extracted.purchase_time == "last time":
        # filter orders for this customer
        customer_orders = [o for o in orders if o["name"].lower() == (extracted.name or "").lower()]
        if customer_orders:
            # sort by date descending
            latest = max(customer_orders, key=lambda o: o["date"])
            sku, date = latest["sku"], latest["date"]
            stock_qty = inventory.get(sku)
            return Retrieval(last_sku=sku, last_date=date, stock_sku=sku, stock_qty=stock_qty)

    # otherwise, fall back to LLM retrieval
    content = call_llama([{"role":"system","content":SYSTEM_RETRIEVE},
                          {"role":"user","content":USER_RETRIEVE(asdict(extracted), orders, inventory)}])
    # print("Retrieval LLM output:", content)
    data = parse_json_from_text(content)
    stock_qty = data.get("stock_qty")
    if isinstance(stock_qty, str) and stock_qty.isdigit():
        stock_qty = int(stock_qty)
    return Retrieval(
        last_sku=data.get("last_sku"),
        last_date=data.get("last_date"),
        stock_sku=data.get("stock_sku"),
        stock_qty=stock_qty if isinstance(stock_qty, int) else None
    )


def output_llama(extracted: Extracted, retrieval: Retrieval, enough: Optional[bool]) -> str:
    comparison = {"requested": extracted.number, "available": retrieval.stock_qty, "enough": enough}
    content = call_llama([{"role":"system","content":SYSTEM_OUTPUT},
                          {"role":"user","content":USER_OUTPUT(asdict(extracted), asdict(retrieval), comparison)}])
    return " ".join(content.strip().split())

def run_pipeline_llama(user_text: str, orders_db: List[str], inventory_db: List[str]) -> str:
    # Parse DBs
    orders = []
    for line in orders_db:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            orders.append({"name":parts[0],"sku":parts[1],"date":parts[2]})
    inventory: Dict[str,int] = {}
    for line in inventory_db:
        parts = [p.strip() for p in line.split(",")]
        if len(parts)>=2:
            try: inventory[parts[0]] = int(parts[1])
            except: pass

    # Steps
    ext = extract_llama(user_text)
    ret = retrieve_llama(ext, orders, inventory)
    enough = None
    if ext.number is not None and ret.stock_qty is not None:
        enough = ret.stock_qty >= ext.number
        # print(f"Requested {ext.number}, available {ret.stock_qty}, enough: {enough}")
    return output_llama(ext, ret, enough)
