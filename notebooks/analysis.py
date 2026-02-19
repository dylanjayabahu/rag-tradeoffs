import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('experiments/pareto_data.csv')

# --- 1. DATA CLEANING & MAPPING ---
pos_map = {'top': 10, 'middle': 50, 'bottom': 90}

def clean_needle_pos(val):
    if isinstance(val, str):
        val = val.lower().strip()
        if val in pos_map:
            return pos_map[val]
    try:
        return float(val)
    except:
        return np.nan

df['needle_pos'] = df['needle_pos'].apply(clean_needle_pos)
df['latency'] = pd.to_numeric(df['latency'], errors='coerce')
df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
df['context_len'] = pd.to_numeric(df['context_len'], errors='coerce')
df = df.dropna(subset=['latency', 'accuracy'])

# Bins for heatmap
df['pos_bin'] = pd.cut(df['needle_pos'], bins=[0, 33, 66, 100], 
                       labels=['Top (0-33%)', 'Middle (33-66%)', 'Bottom (66-100%)'])

df['context_budget'] = df['chunk_size'] * df['top_k']
sns.set_theme(style="whitegrid")

# --- 2. SCATTER PLOT (Accuracy vs Latency) ---
plt.figure(figsize=(12, 6))
model_stats = df.groupby('model').agg({'latency': 'mean', 'accuracy': 'mean'}).reset_index()
sns.scatterplot(data=model_stats, x='latency', y='accuracy', hue='model', s=200)
plt.title('Model Tradeoffs: Accuracy vs. Latency')
plt.xlabel('Mean Latency (s)')
plt.ylabel('Mean Accuracy')
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('experiments/01_pareto_frontier.png', bbox_inches='tight')

# --- 3. HEATMAP (Context Sensitivity) ---
plt.figure(figsize=(14, 8))
pivot_accuracy = df.pivot_table(index='model', columns='pos_bin', 
                               values='accuracy', aggfunc='mean', observed=False)
if not pivot_accuracy.empty:
    sns.heatmap(pivot_accuracy, annot=True, cmap='RdYlGn', fmt=".2f", cbar_kws={'shrink': .8})
    plt.title('Retrieval Accuracy by Position')
    plt.savefig('experiments/02_heatmap_sensitivity.png', bbox_inches='tight')

# --- 4. FAILURE MODE: Accuracy vs. Context Length (The "Cliff") ---
plt.figure(figsize=(12, 6))
# Using lineplot to see how accuracy drops as document length increases
sns.lineplot(data=df, x='context_len', y='accuracy', hue='chunk_size', marker='o', palette='flare')
plt.title('Failure Mode: Accuracy vs. Document Context Length')
plt.xlabel('Context Length (Tokens)')
plt.ylabel('Retrieval Accuracy')
plt.legend(title='Chunk Size', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('experiments/03_failure_modes.png', bbox_inches='tight')

# --- 5. LATENCY SCALING ---
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='context_budget', y='latency', hue='model', marker='o')
plt.title('Latency Scaling by Context Volume')
plt.xlabel('Total Tokens (Chunk Size * Top-K)')
plt.ylabel('Latency (s)')
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('experiments/04_latency_scaling.png', bbox_inches='tight')

# plt.show()