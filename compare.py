import argparse
import torch
from pathlib import Path
from datasets import load_from_disk
from PIL import Image
from unsloth import FastVisionModel
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    # Load dataset
    ds = load_from_disk(args.dataset_dir)
    print(f"Loaded {len(ds)} samples.")

    # Load base model & processor via standard Transformers
    print("Loading base model...")
    base_model, tokenizer = FastVisionModel.from_pretrained(
        args.model_id,
        load_in_4bit=False,
    )
    base_model.to("cuda")
    FastVisionModel.for_inference(base_model)

    # Load fine-tuned model (base + adapter)
    print("Loading fine-tuned model...")
    ft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    ft_model.to("cuda")
    FastVisionModel.for_inference(ft_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx in range(min(args.num_samples, len(ds))):
        row = ds[idx]
        meta = row.get("meta", {})
        question = meta.get("question", "No question")
        gold_answer = meta.get("answer", "No answer")
        
        img_data = row["messages"][1]["content"][0]["image"]
        if isinstance(img_data, dict) and "bytes" in img_data:
            from io import BytesIO
            image = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
        else:
            image = img_data

        # Prepare inputs
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a precise document visual question answering assistant. Read the provided document image carefully and answer the user's question using only information visible in the image. Respond with a concise factual answer. Do not invent details."}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(images=image, text=input_text, add_special_tokens=False, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference Base Model (disable adapter)
        with ft_model.disable_adapter():
            with torch.inference_mode():
                outputs = base_model.generate(**inputs, max_new_tokens=64, do_sample=False)
                base_pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Inference Fine-Tuned Model (enable adapter)
        with torch.inference_mode():
            outputs = ft_model.generate(**inputs, max_new_tokens=64, do_sample=False)
            ft_pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"\n--- Sample {idx+1} ---")
        print(f"Question:  {question}")
        print(f"Gold Ans:  {gold_answer}")
        print(f"Base Pred: {base_pred.strip()}")
        print(f"FT Pred:   {ft_pred.strip()}")

if __name__ == "__main__":
    main()
