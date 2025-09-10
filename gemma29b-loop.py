import os
import time
from transformers import pipeline
import torch

# Initialize the model
model_id = "google/gemma-2-9b"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

# Define directories
QUERY_DIR = "query"
ANSWER_DIR = "answer"

# Ensure directories exist
os.makedirs(QUERY_DIR, exist_ok=True)
os.makedirs(ANSWER_DIR, exist_ok=True)

while True:
    # Scan for .txt files in query directory
    for filename in os.listdir(QUERY_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(QUERY_DIR, filename)
            
            # Read the query from the file
            with open(file_path, "r", encoding="utf-8") as f:
                query = f.read().strip()
            
            # Prepare the message for the pipeline
            messages = [{"role": "user", "content": query}]
            
            # Generate response
            outputs = pipe(
                messages,
                max_new_tokens=131072,
            )
            response = outputs[0]["generated_text"][-1]["content"]
            
            # Create output file path
            output_path = os.path.join(ANSWER_DIR, filename)
            
            # Write query and response to the answer file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Query:\n{query}\n\nAnswer:\n{response}")
            
            # Remove the original query file
            os.remove(file_path)
    
    # Sleep briefly to avoid excessive CPU usage
    time.sleep(1)
