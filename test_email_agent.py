from email_agent_llama_feedback import run_pipeline_llama

orders = [
    "Alex, Cabernet Sauvignon (Barossa) 2021, 2025-09-29",
    "Alex, Prosecco DOC (Sparkling) NV, 2025-07-05",
    "David, Shiraz (McLaren Vale) 2020, 2025-09-29",
    "David, Chardonnay (Yarra Valley) 2024, 2025-05-29",
    "David, Prosecco DOC (Sparkling) NV, 2025-04-29",
]

inventory = [
    "Cabernet Sauvignon (Barossa) 2021, 8",
    "Prosecco DOC (Sparkling) NV, 2",          # low: good for shortage demo
    "Shiraz (McLaren Vale) 2020, 3",
    "Syrah (McLaren Vale) 2020, 7",           # synonym for Shiraz (RAG substitution)
    "Chardonnay (Yarra Valley) 2024, 10",
    "Cava Brut (Sparkling) NV, 12",           # alt sparkling (RAG substitution)
    "Sauvignon Blanc (Marlborough) 2023, 6",
]

examples = [
    # 1. Exact repeat + enough stock
    "Hi, I'm Alex, can I order 2 wines like last time?",
    
    # 2. Fuzzy style + shortage triggers substitution
    "Alex here — please send me 3 bubbly bottles again.",
    
    # 3. Synonym resolution (Syrah ↔ Shiraz)
    "Hello, this is David. I'd like 2 Syrah from last time.",
    
    # 4. Region/style query without SKU name
    "Could you ship 1 dry red from Barossa, same as before?",
    
    # 5. Insufficient stock explicit request
    "Good afternoon, this is Alex. I’d like 5 bottles of Prosecco again.",
]

for i, email in enumerate(examples, 1):
    print("\n============================")
    print(f"Example {i}: {email}")
    print("----------------------------")
    print(run_pipeline_llama(email, orders, inventory))
