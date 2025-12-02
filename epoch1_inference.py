import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

test_json = "prj/pmc_vqa_test_llava.json"
image_folder = "/home/devanggupta/prj/PMC-VQA/images"

model_path = "/home/devanggupta/llava-v1.5-7b-task-lora"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_name="llava-v1.5-7b-lora_epoch1",
    model_path=model_path,
    model_base="liuhaotian/llava-v1.5-7b",
    load_8bit=False,
    load_4bit=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class PMCVQADataset(Dataset):
    def __init__(self, json_path, image_root):
        self.items = json.load(open(json_path))
        self.image_root = image_root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        img_path = os.path.join(self.image_root, item["image"])
        image = Image.open(img_path).convert("RGB")

        # conversation format: [0]=system, [1]=human (<image>\nQ), [2]=gpt
        question = item["conversations"][0]["value"]
        gt_answer = item["conversations"][1]["value"]

        # You can include system prompt in the text if you want:
        # prompt = system_prompt + "\n" + question
        prompt = question

        return {
            "id": item.get("id", idx),
            "image": image,
            "prompt": prompt,
            "gt_answer": gt_answer,
            "raw_item": item,
        }


def collate_fn(batch):
    # images to tensor
    images = [b["image"] for b in batch]
    images = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].half()

    prompts = [b["prompt"] for b in batch]
    input_ids = [tokenizer_image_token(p, tokenizer, return_tensors="pt") for p in prompts]
    # pad to same length
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    return {
        "ids": [b["id"] for b in batch],
        "images": images,
        "input_ids": input_ids,
        "gt_answers": [b["gt_answer"] for b in batch],
        "raw_items": [b["raw_item"] for b in batch],
    }


dataset = PMCVQADataset(test_json, image_folder)
loader = DataLoader(
    dataset,
    batch_size=4,              # tune this based on VRAM
    shuffle=True,
    num_workers=6,
    collate_fn=collate_fn,
)

results = []

model.eval()
with torch.no_grad():
    ctr  = 1 
    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)

        output_ids = model.generate(
            input_ids,
            images=images,
            max_new_tokens=20,
            min_new_tokens=5,
            do_sample=False,
            temperature=0.0,
        )

        for i, out in enumerate(output_ids):
            # pred = tokenizer.decode(out, skip_special_tokens=True).strip()
            raw_pred = tokenizer.decode(out, skip_special_tokens=True).strip()

            # Get the original human message (with <image>\n)
            human_msg = batch["raw_items"][i]["conversations"][0]["value"]

            # Strip off the <image> part to get just the text question
            # e.g. "<image>\nWhat muscle is this?" -> "What muscle is this?"
            if "\n" in human_msg:
                _, plain_q = human_msg.split("\n", 1)
            else:
                plain_q = human_msg
            plain_q = plain_q.strip()

            pred = raw_pred

            # 1) If prediction literally starts with the question text, strip it
            if plain_q and pred.lower().startswith(plain_q.lower()):
                pred = pred[len(plain_q):].strip()

            # 2) If there's still a question mark early on, drop everything up to '?'
            #    (handles cases like "What muscle is this? Vastus lateralis")
            qpos = pred.find("?")
            if qpos != -1 and qpos < 80:   # 80 is just a heuristic cutoff
                pred = pred[qpos+1:].strip()

            # 3) Take only the first line
            # pred = pred.split("\n")[0].strip()

            # # 4) Optional: strip "Answer:" prefix
            # if pred.lower().startswith("answer:"):
            #     pred = pred.split(":", 1)[1].strip()

            results.append({
                "id": batch["ids"][i],
                "image": batch["raw_items"][i]["image"],
                "question": batch["raw_items"][i]["conversations"][0]["value"],
                "prediction": pred,
                "gt_answer": batch["gt_answers"][i].strip(),
            })
            print(f"SAMPLE{ctr} prediction", pred, "gt:", batch["gt_answers"][i].strip())
            ctr  += 1

out_path = "pmc_vqa_llava_epoch1_predictions.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved predictions to {out_path}")
