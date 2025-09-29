from email_agent_llama import run_pipeline_llama

orders = [
    "Alex, wine A, 2025-09-29",
    "Alex, wine B, 2025-07-05",
    "David, wine B, 2025-09-29",    
    "David, wine A, 2025-05-29",
]
inventory = ["wine A, 5", "wine B, 3"]

# email = "Hi, I'm Alex, I want to order 2 wines I ordered last time."
# print("Input email:\n", email)
# print("Output response:")
# print(run_pipeline_llama(email, orders, inventory))

# print("\n---\n")

# email = "Hello, this is David. I’d like to order the same two wines I purchased last time."
# print("Input email:\n", email)
# print("Output response:")
# print(run_pipeline_llama(email, orders, inventory))

# print("\n---\n")

email = "Good afternoon, my name is David. I’d like to order the same five bottles of wine that I got during my last order, as I’d like to enjoy them again."
print("Input email:\n", email)
print("Output response:")
print(run_pipeline_llama(email, orders, inventory))

print("\n---\n")