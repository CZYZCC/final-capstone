import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_theme(style="whitegrid")

# Output directory for results
OUTPUT_DIR = "experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input files mapping
FILES = {
    "no_retrieval": "experiment_data/scored_no_retrieval.jsonl",
    "vector_rag": "experiment_data/scored_vector_rag.jsonl",
    "graph_rag": "experiment_data/scored_graph_rag.jsonl"
}

def load_distribution_data():
    """Load only the necessary metadata for distribution analysis."""
    records = []
    for model_name, filepath in FILES.items():
        if not os.path.exists(filepath):
            print(f"[Warning] File not found: {filepath}")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    meta = data.get("metadata", {})
                    score_data = data.get("score", {})
                    
                    # Skip items that failed the evaluation step
                    if "error" in score_data:
                        continue
                        
                    record = {
                        "q_type": meta.get("expected_type", "unknown"),
                        "q_format": meta.get("question_format", "unknown")
                    }
                    records.append(record)
                except json.JSONDecodeError:
                    continue
                    
    return pd.DataFrame(records)

def plot_separate_pie_charts(df):
    """Generate and save separate pie charts for question types and formats."""
    if df.empty:
        print("[!] No data available to plot.")
        return

    # ---------------------------------------------------------
    # Chart 1: Question Type (Conceptual vs Computational)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    type_counts = df['q_type'].value_counts()
    
    plt.pie(
        type_counts, 
        labels=type_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=sns.color_palette("pastel"),
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 12}
    )
    plt.title('Distribution of Question Types\n(Conceptual vs Computational)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    path_type = os.path.join(OUTPUT_DIR, "distribution_question_type.png")
    plt.savefig(path_type, dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Chart 2: Question Format (MCQ, True/False, etc.)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    format_counts = df['q_format'].value_counts()
    
    plt.pie(
        format_counts, 
        labels=format_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=sns.color_palette("Set3"),
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 12}
    )
    plt.title('Distribution of Question Formats', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    path_format = os.path.join(OUTPUT_DIR, "distribution_question_format.png")
    plt.savefig(path_format, dpi=300)
    plt.close()
    
    print(f"[*] Pie charts successfully saved to:")
    print(f"    1. {path_type}")
    print(f"    2. {path_format}")

if __name__ == "__main__":
    print("[*] Loading data for dataset distribution analysis...")
    df = load_distribution_data()
    
    if df.empty:
        print("[!] Analysis aborted due to missing data.")
    else:
        plot_separate_pie_charts(df)