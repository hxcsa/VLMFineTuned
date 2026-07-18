import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import load_from_disk
from PIL import Image

def main():
    ds = load_from_disk("/workspace/VLMFineTuned/data/docvqa_test")
    row = ds[0]
    question = row["meta"]["question"]
    gold = row["meta"]["answer"]
    
    # Load PIL image
    img_data = row["messages"][1]["content"][0]["image"]
    if isinstance(img_data, dict) and "bytes" in img_data:
        from io import BytesIO
        image = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
    else:
        image = img_data

    print(f"Question: {question}")
    print(f"Gold Answer: {gold}")

    print("Loading model and processor via HF transformers...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # Format messages
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a precise document visual question answering assistant."}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f"HF Base Pred: {output_text[0]}")

if __name__ == "__main__":
    main()
