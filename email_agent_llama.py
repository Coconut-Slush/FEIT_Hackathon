import os, json, re, requests
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from typing import Literal  # add to your imports

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

# -------- Patch normalization & safe access helpers --------
def _coalesce(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default

def normalize_patch(p: dict) -> dict | None:
    """Return a normalized patch or None if invalid."""
    if not isinstance(p, dict) or "type" not in p:
        return None
    t = p.get("type")

    if t == "synonym":
        # Accept multiple historic schemas
        pattern = _coalesce(p, "pattern", "term", "source", "from")
        maps_to = _coalesce(p, "maps_to", "synonym", "target", "to")
        if not pattern or not maps_to:
            return None
        return {"type": "synonym", "pattern": str(pattern), "maps_to": str(maps_to)}

    if t == "substitution":
        if_req = _coalesce(p, "if_request", "sku", "item", "when_requesting")
        when_below = _coalesce(p, "when_stock_below", "below", "threshold")
        suggest = _coalesce(p, "suggest", "replace_with", "offer")
        try:
            when_below = int(when_below) if when_below is not None else None
        except Exception:
            return None
        if not if_req or when_below is None or not suggest:
            return None
        return {
            "type": "substitution",
            "if_request": str(if_req),
            "when_stock_below": int(when_below),
            "suggest": str(suggest),
        }

    if t == "fact":
        key = _coalesce(p, "key")
        val = _coalesce(p, "value", "val")
        if key is None or val is None:
            return None
        return {"type": "fact", "key": str(key), "value": str(val)}

    # Unknown type
    return None


FEEDBACK_DB_PATH = "feedback_db.json"


def load_feedback_db() -> dict:
    try:
        with open(FEEDBACK_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
    except Exception:
        return {"rules": [], "facts": []}

    # Defensive structure
    rules = db.get("rules", []) or []
    facts = db.get("facts", []) or []

    # Normalize everything we find; drop invalid entries
    norm_rules = []
    for r in rules:
        nr = normalize_patch(r)
        if nr:
            # facts sometimes ended up in "rules" historically; redirect them
            if nr["type"] == "fact":
                facts.append({"type": "fact", "key": nr["key"], "value": nr["value"], "source": r.get("source", "owner")})
            else:
                nr["source"] = r.get("source", "owner")
                norm_rules.append(nr)

    norm_facts = []
    for fa in facts:
        nfa = normalize_patch(fa)
        if nfa and nfa["type"] == "fact":
            nfa["source"] = fa.get("source", "owner")
            norm_facts.append(nfa)

    return {"rules": norm_rules, "facts": norm_facts}


def save_feedback_db(db: dict) -> None:
    with open(FEEDBACK_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


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
# def call_llama(messages, model="llama-3.3-70b-versatile", temperature=0.0):
#     """Call Groq (OpenAI-compatible) with a Llama-3 model."""
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise RuntimeError("Missing GROQ_API_KEY")
#     url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     payload = {"model": model, "messages": messages, "temperature": temperature}
#     resp = requests.post(url, headers=headers, json=payload, timeout=60)
#     resp.raise_for_status()
#     return resp.json()["choices"][0]["message"]["content"]

def call_llama(messages, temperature: float = 0.0, timeout: int = 60):
    """
    Minimal Groq caller:
      - fixed model: llama-3.3-70b-versatile
      - no retries/backoff
      - respects a temperature arg
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": float(temperature),
        # keep defaults simple; no seed/top_p unless you want them
        "n": 1,
        "stream": False,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ----------------------- Prompts -----------------------
SYSTEM_EXTRACT = """You are an information extraction model.
Return STRICT JSON with keys: name, number, type, purchase_time.
- name: string or null
- number: integer or null
- type: string or null.
  IMPORTANT: If the text mentions only a style or shorthand (e.g., "Prosecco", "Cabernet"),
  expand it to the closest matching SKU from the provided inventory, e.g., "Prosecco DOC (Sparkling) NV".
  Do not output vague words like "wine".
  If the email says "same as last time" or equivalent, set type = null.
- purchase_time: string or null (use "last time" if explicitly stated; otherwise null is fine).
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
You will receive a 'context' block (RAG CONTEXT) with short facts from a knowledge base.
Use those facts plus the structured fields to write ONE clear paragraph.
If stock is sufficient: confirm and recommend proceeding.
If not: offer a close substitute (respect style/synonyms) or partial+backorder with ETA.
Do NOT output JSON or bullets; write one coherent message.
"""
def USER_OUTPUT(extracted, retrieval, comparison, context: str = ""):
    return json.dumps({
        "context": context,               # <= NEW: pass RAG context to the writer
        "extracted": extracted,
        "retrieval": retrieval,
        "comparison": comparison
    }, ensure_ascii=False)
    
SYSTEM_FEEDBACK = """You convert an owner's short comment into ONE typed JSON patch.
Allowed:
- {"type":"synonym","pattern":"bubbly","maps_to":"sparkling"}
- {"type":"substitution","if_request":"Prosecco DOC (Sparkling) NV","when_stock_below":3,"suggest":"Cava Brut (Sparkling) NV"}
- {"type":"fact","key":"Chardonnay (Yarra Valley) 2024:style","value":"white"}
Return ONLY the JSON object."""

def owner_feedback_to_patch(feedback_text: str) -> Optional[dict]:
    content = call_llama(
        [{"role":"system","content":SYSTEM_FEEDBACK},
         {"role":"user","content":feedback_text.strip()}],
        temperature=0.0
    )
    try:
        patch = parse_json_from_text(content)
        return patch if isinstance(patch, dict) and "type" in patch else None
    except Exception:
        return None

SYSTEM_CRITIC = """You are a strict plan critic for an order-handling agent.
Input JSON keys: extracted, retrieval, comparison, context, response.
Tasks:
1) Detect likely issues (synonym/style mismatch, substitution rule opportunity, fact fix).
2) Propose minimal kb_patches using the same schema as the feedback converter.
Return ONLY JSON: {"conflicts":[], "improvements":[], "kb_patches":[], "action":"ask"}.
"""

def critic_llama(payload: dict) -> dict:
    content = call_llama(
        [{"role":"system","content":SYSTEM_CRITIC},
         {"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
        temperature=0.0
    )
    try:
        return parse_json_from_text(content)
    except Exception:
        return {"conflicts": [], "improvements": [], "kb_patches": [], "action": "ask"}

def format_kb_suggestions(critic_report: dict) -> str:
    raw = critic_report.get("kb_patches", [])
    if not isinstance(raw, list) or not raw:
        return "No suggested knowledge updates."

    lines = ["Suggested updates:"]
    for p in raw:
        np = normalize_patch(p)
        if not np:
            continue
        t = np["type"]
        if t == "synonym":
            lines.append(f"- Synonym: '{np['pattern']}' → '{np['maps_to']}'")
        elif t == "substitution":
            lines.append(
                f"- Substitution: If '{np['if_request']}' below {np['when_stock_below']}, suggest '{np['suggest']}'"
            )
        elif t == "fact":
            lines.append(f"- Fact: {np['key']} = {np['value']}")
    return "\n".join(lines) if len(lines) > 1 else "No suggested knowledge updates."


# ===== RAG helpers (dependency-free) =====
import math

def _lexical_score(q: str, doc: str) -> float:
    q_terms = set(re.findall(r"\w+", (q or "").lower()))
    d_terms = set(re.findall(r"\w+", (doc or "").lower()))
    if not q_terms or not d_terms: 
        return 0.0
    # Simple Jaccard-ish score: robust & fast for a demo
    return len(q_terms & d_terms) / max(1, len(q_terms | d_terms))

def top_k_context(query: str, docs: list[str], k: int = 4) -> list[tuple[float,str]]:
    scored = [(_lexical_score(query, d), d) for d in docs]
    return sorted(scored, key=lambda x: x[0], reverse=True)[:k]

def build_context_block(items: list[tuple[float,str]]) -> str:
    if not items: 
        return "RAG CONTEXT:\n(none)"
    lines = [f"- {doc}" for _, doc in items]
    return "RAG CONTEXT:\n" + "\n".join(lines)

def build_rag_docs(ext: Extracted, orders: list[dict], inventory: dict[str,int]) -> list[str]:
    docs: list[str] = []

    # # (NEW) Owner feedback first → highest precedence
    # fdb = load_feedback_db()
    # for r in fdb.get("rules", []):
    #     if r.get("type") == "synonym":
    #         docs.append(f"Synonym: {r['pattern']} means {r['maps_to']}.")
    #     elif r.get("type") == "substitution":
    #         docs.append(f"Rule: If requesting {r['if_request']} and stock below {r['when_stock_below']}, suggest {r['suggest']}.")
    # for fact in fdb.get("facts", []):
    #     docs.append(f"Fact: {fact['key']} = {fact['value']}.")
    
    # (NEW) Owner feedback first → highest precedence
    fdb = load_feedback_db()
    for r in fdb.get("rules", []):
        nr = normalize_patch(r)
        if not nr:
            continue
        if nr["type"] == "synonym":
            docs.append(f"Synonym: {nr['pattern']} means {nr['maps_to']}.")
        elif nr["type"] == "substitution":
            docs.append(
                f"Rule: If requesting {nr['if_request']} and stock below {nr['when_stock_below']}, suggest {nr['suggest']}."
            )
    for fact in fdb.get("facts", []):
        nfa = normalize_patch(fact)
        if nfa and nfa["type"] == "fact":
            docs.append(f"Fact: {nfa['key']} = {nfa['value']}.")


    # (existing code continues: history, inventory, curated knowledge)
    # 1) Customer history
    if ext and ext.name:
        for o in orders:
            if o["name"].lower() == (ext.name or "").lower():
                docs.append(f"History: {o['name']} bought {o['sku']} on {o['date']}.")
    # 2) Inventory
    for sku, qty in inventory.items():
        docs.append(f"Stock: {sku} has {qty} units available.")
    # 3) Your curated snippets (leave as-is)
    docs += [
        "Synonym: Shiraz is also called Syrah in many regions.",
        "Style: Cabernet Sauvignon (Barossa) 2021 is a dry red.",
        "Style: Shiraz (McLaren Vale) 2020 is a dry red; Syrah (McLaren Vale) 2020 is equivalent style.",
        "Style: Prosecco DOC (Sparkling) NV and Cava Brut (Sparkling) NV are sparkling wines.",
        "Rule: If sparkling is requested and Prosecco is low, suggest Cava Brut as a substitute.",
        "Rule: If user says 'last time', prefer exact last-SKU; otherwise match by style and availability.",
    ]
    return docs

# === Learned-rule application helpers ===

def _best_inventory_key(query: str, inventory: dict[str, int]) -> Optional[str]:
    """Map a partial or informal name to a concrete SKU key in inventory."""
    if not query:
        return None
    # 1) Exact match
    for k in inventory.keys():
        if k == query:
            return k
    # 2) Case-insensitive exact
    ql = query.lower()
    for k in inventory.keys():
        if k.lower() == ql:
            return k
    # 3) Substring match
    subs = [k for k in inventory.keys() if ql in k.lower()]
    if subs:
        return subs[0]
    # 4) Fallback: lexical score
    scored = [(_lexical_score(query, k), k) for k in inventory.keys()]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def apply_learned_rules(
    ext: Extracted,
    ret: Retrieval,
    inventory: dict[str, int]
) -> tuple[Retrieval, Optional[dict]]:
    """
    Deterministically apply learned substitution rules:
      - If a rule matches the requested item and its stock is below threshold,
        swap to the suggested SKU and update stock_qty.
    Returns (updated_retrieval, info_or_None).
    """
    db = load_feedback_db()
    requested_text = (ret.stock_sku or ext.type or "").strip()
    if not requested_text:
        return ret, None

    # Canonicalize what the user is essentially asking for
    req_canon = _best_inventory_key(requested_text, inventory) or ret.stock_sku

    for r in db.get("rules", []):
        nr = normalize_patch(r)
        if not nr or nr["type"] != "substitution":
            continue

        rule_req = nr["if_request"]
        src_canon = _best_inventory_key(rule_req, inventory)

        # Consider as matched if:
        #   - canonical SKUs match, OR
        #   - the human text of the rule is a substring of the requested text or canonical SKU
        match = False
        if req_canon and src_canon and req_canon.lower() == src_canon.lower():
            match = True
        elif rule_req.lower() in requested_text.lower():
            match = True
        elif req_canon and rule_req.lower() in req_canon.lower():
            match = True

        if not match:
            continue

        # What is the stock of the *source* item we’re substituting away from?
        src_key = src_canon or req_canon
        src_qty = inventory.get(src_key)

        # Fallback to whatever retrieval found if direct lookup failed
        if src_qty is None and ret.stock_sku:
            src_qty = inventory.get(ret.stock_sku, ret.stock_qty)

        # Apply threshold
        threshold = int(nr["when_stock_below"])
        if src_qty is not None and src_qty < threshold:
            # Choose the concrete suggestion key
            sug_canon = _best_inventory_key(nr["suggest"], inventory)
            ret.stock_sku = sug_canon or nr["suggest"]
            ret.stock_qty = inventory.get(sug_canon) if sug_canon else None

            return ret, {
                "applied_rule": nr,
                "src_key": src_key,
                "src_qty": src_qty,
                "sug_key": ret.stock_sku,
                "sug_qty": ret.stock_qty,
            }

    return ret, None


# ----------------------- Pipeline -----------------------
def extract_llama(text: str) -> Extracted:
    content = call_llama([{"role":"system","content":SYSTEM_EXTRACT},
                          {"role":"user","content":USER_EXTRACT(text)}])
    # print("Extraction LLM output:", content)
    data = parse_json_from_text(content)
    return Extracted(**data)

def retrieve_llama(extracted: Extracted, orders: List[Dict[str,str]], inventory: Dict[str,int]) -> Retrieval:
    """
    Resolution order:
    1) If user implies 'last time' (explicit or implicit via no type+no time),
       pick the latest order for that customer and use it as both last_* and stock_*.
    2) If user explicitly requests a SKU (or style that we can normalize to a SKU),
       set stock_* to that SKU and set last_* to the customer's most recent order of that same SKU (if any).
    3) Otherwise, fall back to LLM retrieval.
    """

    # helpers
    def _latest_order_for_customer(name: Optional[str]) -> Optional[Dict[str, str]]:
        if not name:
            return None
        cust_orders = [o for o in orders if o["name"].lower() == name.lower()]
        if not cust_orders:
            return None
        return max(cust_orders, key=lambda o: o["date"])  # YYYY-MM-DD sortable

    def _latest_order_for_customer_sku(name: Optional[str], sku: str) -> Optional[Dict[str, str]]:
        if not name or not sku:
            return None
        sku_l = sku.lower()
        cust_orders = [
            o for o in orders
            if o["name"].lower() == name.lower() and o["sku"].lower() == sku_l
        ]
        if not cust_orders:
            return None
        return max(cust_orders, key=lambda o: o["date"])

    # ---- Determine if "last time" is implied (same logic as before) ----
    implied_last_time = (extracted.purchase_time == "last time") or (
        (not extracted.purchase_time) and (not extracted.type)
    )

    # ---- Fast path: "same as last time" with no explicit type ----
    if implied_last_time:
        latest = _latest_order_for_customer(extracted.name)
        if latest:
            sku, date = latest["sku"], latest["date"]
            stock_qty = inventory.get(sku)
            return Retrieval(last_sku=sku, last_date=date, stock_sku=sku, stock_qty=stock_qty)

    # ---- If the user gave a type, normalize it to a concrete SKU and prefer it ----
    requested_sku = None
    if extracted.type:
        requested_sku = _best_inventory_key(extracted.type, inventory)

    if requested_sku:
        # Use the requested SKU for stock, and look up the most recent matching order for "last_*"
        last_match = _latest_order_for_customer_sku(extracted.name, requested_sku)
        last_sku = last_match["sku"] if last_match else None
        last_date = last_match["date"] if last_match else None
        stock_qty = inventory.get(requested_sku)
        return Retrieval(last_sku=last_sku, last_date=last_date, stock_sku=requested_sku, stock_qty=stock_qty)

    # ---- Otherwise fall back to LLM retrieval (unchanged) ----
    content = call_llama(
        [
            {"role":"system","content":SYSTEM_RETRIEVE},
            {"role":"user","content":USER_RETRIEVE(asdict(extracted), orders, inventory)}
        ]
    )
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

def output_llama(extracted: Extracted, retrieval: Retrieval, enough: Optional[bool], extra_context: str = "") -> str:
    comparison = {"requested": extracted.number, "available": retrieval.stock_qty, "enough": enough}
    content = call_llama(
        [
            {"role":"system","content":SYSTEM_OUTPUT},
            {"role":"user","content":USER_OUTPUT(asdict(extracted), asdict(retrieval), comparison, extra_context)}
        ]
    )
    return " ".join(content.strip().split())


# CHANGE SIGNATURE:
def run_pipeline_llama(user_text: str,
                       orders_db: List[str],
                       inventory_db: List[str],
                       owner_decision: Literal["accept","reject"] = "accept",
                       owner_comment: Optional[str] = None) -> str:
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

    # NEW: deterministically apply learned substitution rules
    ret, applied = apply_learned_rules(ext, ret, inventory)

    # Stock sufficiency (recompute after possible substitution)
    enough = None
    if ext.number is not None and ret.stock_qty is not None:
        enough = ret.stock_qty >= ext.number


    # ===== NEW: RAG context =====
    docs = build_rag_docs(ext, orders, inventory)
    query = f"{ext.name or ''} {ext.type or ''} last:{ret.last_sku or ''} style:{('sparkling' if 'sparkling' in (ret.last_sku or '').lower() else '')}"
    ctx_block = build_context_block(top_k_context(query=query, docs=docs, k=4))

    # print(ctx_block)

    response = output_llama(ext, ret, enough, extra_context=ctx_block)

    # (NEW) Always prepare critic suggestions (non-binding)
    critic_report = critic_llama({
        "extracted": asdict(ext),
        "retrieval": asdict(ret),
        "comparison": {"requested": ext.number, "available": ret.stock_qty, "enough": bool(enough)},
        "context": ctx_block,
        "response": response
    })
    # suggestions_text = format_kb_suggestions(critic_report)

    # (NEW) Owner-in-the-loop policy
    if owner_decision == "reject":
        if owner_comment and owner_comment.strip():
            # Learn ONLY from owner's comment
            print("\nOwner comment provided; attempting to learn...")
            patch = owner_feedback_to_patch(owner_comment)
            norm = normalize_patch(patch) if patch else None
            if norm:
                db = load_feedback_db()
                if norm["type"] == "fact":
                    db["facts"].append({**norm, "source": "owner"})
                else:
                    db["rules"].append({**norm, "source": "owner"})
                save_feedback_db(db)
                response += " (Update recorded for future runs.)"
        #     else:
        #         # Comment provided but unparsable: show critic suggestions; do NOT update
        #         response += "\n\n" + suggestions_text
        # else:
        #     # Reject with NO comment: do NOT update; only show suggestions
        #     response += "\n\n" + suggestions_text

    return response
