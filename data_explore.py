from datasets import load_dataset
import pandas as pd

dataset = load_dataset("tuetschek/atis")

print("=== Dataset Structure ===")
print(dataset)

print("\n=== Split Sizes ===")
print(f"Train samples : {len(dataset['train'])}")
print(f"Test samples  : {len(dataset['test'])}")

print("\n=== First 3 Examples ===")
for i in range(3):
    sample = dataset['train'][i]
    print(f"\n[{i}] text   : {sample['text']}")
    print(f"    intent : {sample['intent']}")

print("\n=== All Intent Classes ===")
intents = dataset['train']['intent']
unique_intents = sorted(set(intents))
for idx, name in enumerate(unique_intents):
    print(f"  {idx:2d} → {name}")

print(f"\nTotal intent classes: {len(unique_intents)}")

print("\n=== Class Distribution (Top 10) ===")
df = pd.DataFrame({'intent': intents})
top10 = df['intent'].value_counts().head(10)
print(top10.to_string())