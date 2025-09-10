#!/usr/bin/env python3
import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def run_pipeline_inference(model_name, query, max_tokens=4000):
    """Run inference using the pipeline approach"""
    print(f"\n{'='*80}")
    print(f"Running Pipeline Inference with {model_name}")
    print(f"{'='*80}\n")
    
    pipe = pipeline(
        "text-generation", 
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    result = pipe(
        query,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    print("Pipeline Result:")
    print("-" * 40)
    print(result[0]["generated_text"])
    return result[0]["generated_text"]

def run_direct_inference(model_name, query, max_tokens=4000):
    """Run inference by loading the model directly"""
    print(f"\n{'='*80}")
    print(f"Running Direct Model Inference with {model_name}")
    print(f"{'='*80}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Direct Model Result:")
    print("-" * 40)
    print(generated_text)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference using pipeline and direct model loading")
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/gpt-neox-20b",
        help="Model name from Hugging Face Hub (default: EleutherAI/gpt-neox-20b)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Do you believe in the Big Bang? Explain arguments for and against",
        help="Query to send to the model"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum number of tokens to generate (default: 4000)"
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Run only pipeline inference"
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Run only direct model inference"
    )
    
    args = parser.parse_args()
    
    print(f"Model: {args.model}")
    print(f"Query: {args.query}")
    print(f"Max Tokens: {args.max_tokens}")
    
    # Run pipeline inference
    if not args.direct_only:
        pipeline_result = run_pipeline_inference(args.model, args.query, args.max_tokens)
    
    # Run direct model inference
    if not args.pipeline_only:
        direct_result = run_direct_inference(args.model, args.query, args.max_tokens)
    
    print(f"\n{'='*80}")
    print("Inference Complete")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()