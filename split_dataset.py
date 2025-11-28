import pandas as pd
from sklearn.model_selection import train_test_split


dataset_dir = "/home/devanggupta/prj/PMC-VQA"

train_csv = f"{dataset_dir}/train.csv"
test_csv = f"{dataset_dir}/test.csv"

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

print("Original train size:", len(df_train))
print("Original test size:", len(df_test))


train_final, val_final = train_test_split(
    df_train,
    test_size=0.20,     
    random_state=42,
    shuffle=True
)

print("New train size:", len(train_final))
print("New val size:", len(val_final))


train_final.to_csv(f"{dataset_dir}/train_final.csv", index=False)
val_final.to_csv(f"{dataset_dir}/val_final.csv", index=False)
df_test.to_csv(f"{dataset_dir}/test_final.csv", index=False)

print("Saved:")
print(f"- {dataset_dir}/train_final.csv")
print(f"- {dataset_dir}/val_final.csv")
print(f"- {dataset_dir}/test_final.csv")
