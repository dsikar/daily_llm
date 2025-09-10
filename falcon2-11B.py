from transformers import AutoTokenizer
import transformers
import torch

model = "tiiuae/falcon-11B"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

sequences = pipeline(
   "Can you explain the concept of Quantum Computing?",
    max_length=2000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

