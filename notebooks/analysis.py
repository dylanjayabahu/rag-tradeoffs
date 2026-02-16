import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('experiments/pareto_data.csv')

# 1. Generate Pareto Frontier Data
# Group by model to see the average tradeoff
model_stats = df.groupby('model').agg({
    'latency': 'mean',
    'accuracy': 'mean'
}).reset_index()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=model_stats, x='latency', y='accuracy', hue='model', s=200)

# Add labels for the Pareto Frontier
plt.title('Pareto Frontier: Accuracy vs. Inference Latency')
plt.xlabel('Average Latency (seconds)')
plt.ylabel('Mean Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('experiments/pareto_frontier.png')
plt.show()

# 2. Failure Mode: Accuracy vs. Context Length (The "Cliff")
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='context_len', y='accuracy', hue='chunk_size', marker='o')

plt.title('Failure Mode: Accuracy vs. Document Context Length')
plt.xlabel('Context Length (Tokens)')
plt.ylabel('Retrieval Accuracy')
plt.savefig('experiments/failure_modes.png')
plt.show()