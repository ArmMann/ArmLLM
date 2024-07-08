from datasets import load_dataset

# Load the dataset
dataset = load_dataset("microsoft/cats_vs_dogs", trust_remote_code=True)

print("Dataset structure:", dataset)
