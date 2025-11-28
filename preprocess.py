import pandas as pd
import os
import json

dataset_dir = "/home/devanggupta/prj/PMC-VQA"

train_csv = os.path.join(dataset_dir, "train_final.csv")
val_csv   = os.path.join(dataset_dir, "val_final.csv")
test_csv  = os.path.join(dataset_dir, "test_final.csv")

# train_mod = os.path.join(dataset_dir, "open_train.csv")
# val_mod   = os.path.join(dataset_dir, "open_val.csv")
# test_mod  = os.path.join(dataset_dir, "open_test.csv")

train_llava = os.path.join(dataset_dir, "pmc_vqa_train_llava.json")
val_llava   = os.path.join(dataset_dir, "pmc_vqa_val_llava.json")
test_llava  = os.path.join(dataset_dir, "pmc_vqa_test_llava.json")


def normalize_answer(text):
    if isinstance(text, str):
        text = text.strip().lower()
        # you can add more rules here later if you want
        return text
    return text


def convert_split(in_path, out_path):
    print(f"Loading {in_path}")
    df = pd.read_csv(in_path)

    print("Columns:", df.columns.tolist())

    needed_cols = ["Figure_path", "Question", "Answer"]
    required_cols = ["image_path", "question", "answer"]


    df_open = df[needed_cols].copy()

    df_open = df_open.rename(
        columns={
            "Figure_path": "image_path",
            "Question": "question",
            "Answer": "answer",
        }
    )


    df_open["answer"] = df_open["answer"].apply(normalize_answer)

    for col in required_cols:
        if col not in df_open.columns:
            raise ValueError(f"Expected column '{col}' in {in_path}, got {df_open.columns.tolist()}")

    records = []
    i = 0 
    invalid_answers = [ 
        "none of the above",
        "n/a",
        "not applicable",
        "cannot be determined",
        "cannot determine",
        "cannot be assessed",
        "unknown",
        "not available",
        "not provided",
        "not reported",
        ]
    for idx, row in df_open.iterrows():
        image = str(row["image_path"]).strip()
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        if answer.lower() in invalid_answers or answer == "":
            continue
        sample = {
            "id": int(idx),
            "image": image,
            "conversations": [
                {
                    "from": "human",
                    "value": """ You are a medical visual question answering assistant. A medical image is provided. Answer the question based on the image. Provide only the answer. Do not use full sentences. Do not include explanations\n<image>\n""" + question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        records.append(sample)

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} items to {out_path}")

def main():
    # Step 1 - make open_*.csv
    convert_split(train_csv, train_llava)
    convert_split(val_csv, val_llava)
    convert_split(test_csv, test_llava)

    # Step 2 - convert open_*.csv to LLaVA JSON
    # csv_to_llava_json(train_mod, train_llava)
    # csv_to_llava_json(val_mod, val_llava)
    # csv_to_llava_json(test_mod, test_llava)

    print("Done. Created:")
    print("  ", train_llava)
    print("  ", val_llava)
    print("  ", test_llava)


if __name__ == "__main__":
    main()
