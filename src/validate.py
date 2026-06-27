import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
from pathlib import Path
import json

# --- Configuration ---

DB_PATH = "results/sna.db"
METRICS_JSON = "graphs/metrics_report.json"
OUT_DIR = Path("graphs/validation")

def load_db_predictions():
    print("[1] A ler as previsões reais do sistema na SQLite...")
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT r.id, r.original_text, r.taxonomy_category, MAX(rel.confidence_score) as max_score
        FROM results r
        LEFT JOIN relations rel ON r.id = rel.result_id
        GROUP BY r.id
        ORDER BY r.id ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def run_real_validation():
    # Garantir que a pasta de destino existe
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_preds = load_db_predictions()
    if df_preds.empty:
        print("[!] Erro: A base de dados sna.db está vazia. Corre o ingest.py primeiro!")
        return
        
    print("[2] A descarregar o Ground Truth real do Berkeley dataset...")
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default", split="train")
    df_hf = ds.to_pandas()
    
    # CRITICAL: TEXT ALIGNMENT 
    print("[3] A alinhar os textos da base de dados com as anotações originais...")
    
    # Clean text to ensure exact matching
    df_preds['clean_text'] = df_preds['original_text'].str.strip().str.lower()
    df_hf['clean_text'] = df_hf['text'].str.strip().str.lower()
    
    # Merge on exact text match
    merged = pd.merge(df_preds, df_hf[['clean_text', 'hate_speech_score']], on='clean_text', how='inner')
    
    # Average scores for texts evaluated by multiple annotators
    merged = merged.groupby(['id', 'original_text', 'taxonomy_category', 'max_score'], as_index=False)['hate_speech_score'].mean()
    
    num_samples = len(merged)
    print(f"[*] Sucesso: {num_samples} amostras alinhadas perfeitamente!")
    
    if num_samples == 0:
        print("[!] Erro: Não foi possível alinhar nenhum texto.")
        return
    
    # 1. Define y_true (Berkeley Ground Truth: > 0 indicates hate presence)
    y_true = (merged['hate_speech_score'] > 0.0).astype(int).values
    
    # 2. Define y_pred from model taxonomy
    severe_cats = ["HATE", "THREAT", "RADICALISM", "EXTREMIST"]
    merged['y_pred'] = merged['taxonomy_category'].apply(lambda x: 1 if x in severe_cats else 0)
    
    # 3. Convert confidence score to probability
    merged['y_prob'] = merged['max_score'].fillna(50) / 100.0
    merged.loc[merged['y_pred'] == 0, 'y_prob'] = 1.0 - merged.loc[merged['y_pred'] == 0, 'y_prob']
    
    y_pred = merged['y_pred'].values
    y_prob = merged['y_prob'].values
    
    # CHART 1: REAL CONFUSION MATRIX
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Sem Risco', 'Risco Severo'],
                yticklabels=['Sem Risco', 'Risco Severo'],
                annot_kws={"size": 15, "weight": "bold"})
    plt.title('Matriz de Confusão REAL (Pipeline Híbrido)', pad=15, fontsize=13, fontweight='bold')
    plt.ylabel('Anotadores Humanos (Berkeley)', fontsize=11)
    plt.xlabel('Previsão do LLM', fontsize=11)
    plt.tight_layout()
    # Gravado a 300 DPI para alta resolução (Thesis Quality)
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    
    # CHART 2: REAL CLASS DISTRIBUTION
    
    plt.figure(figsize=(7, 5))
    sns.set_theme(style="whitegrid")
    ax = sns.countplot(x=y_true, hue=y_true, palette=["#74c476", "#d62728"], legend=False)
    plt.title("Distribuição REAL de Classes (Berkeley)", pad=15, fontsize=12, fontweight='bold')
    plt.xlabel("Categoria de Risco", fontsize=11)
    plt.ylabel("Número de Amostras", fontsize=11)
    plt.xticks(ticks=[0, 1], labels=["Sem Risco", "Risco Severo"])
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', 
                    fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=300)
    plt.close()

    # CHART 3: REAL ROC CURVE
    
    fpr_final, tpr_final, _ = roc_curve(y_true, y_prob)
    roc_auc_final = auc(fpr_final, tpr_final)
    
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    plt.plot(fpr_final, tpr_final, color='red', lw=2.5, 
             label=f'Pipeline Híbrido (AUC = {roc_auc_final:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=11)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=11)
    plt.title('Curva ROC Autêntica do Sistema', pad=15, fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    plt.tight_layout()
    # Nome ajustado para sincronizar com a aba de Validação Científica do app.py
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=300)
    plt.close()
    
    # CHART 4: NETWORK METRICS (SNA)
    
    print("[4] A analisar Métricas de Rede (Graph Analytics)...")
    if Path(METRICS_JSON).exists():
        with open(METRICS_JSON, 'r', encoding='utf-8') as f:
            sna_data = json.load(f)
            
        if sna_data:
            sorted_nodes = sorted(sna_data.items(), key=lambda x: x[1].get('eigenvector', 0), reverse=True)[:10]
            nodes = [n[0] for n in sorted_nodes]
            scores = [n[1].get('eigenvector', 0) for n in sorted_nodes]
            
            plt.figure(figsize=(9, 5))
            sns.barplot(x=scores, y=nodes, hue=nodes, palette="magma", legend=False)
            plt.title('Top 10 Entidades Mais Influentes na Rede (PageRank)', pad=15, fontweight='bold')
            plt.xlabel('Score de Influência (PageRank / Eigenvector)')
            plt.ylabel('Entidades / Atores')
            plt.tight_layout()
            plt.savefig(OUT_DIR / "pagerank_top10.png", dpi=300)
            plt.close()
            print("[✓] Gráfico de Análise de Redes (SNA) gerado!")
    else:
        print("[!] analytics.py não foi executado ou metrics_report.json não existe.")

    # METRICS REPORT

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    report_txt = (
        "====================================================\n"
        "     RELATÓRIO DE MÉTRICAS REAIS - PIPELINE SNA     \n"
        "====================================================\n"
        f"Amostras Alinhadas   : {num_samples}\n"
        f"Accuracy  (Exatidão) : {acc:.4f} ({acc*100:.1f}%)\n"
        f"Precision (Precisão) : {prec:.4f} ({prec*100:.1f}%)\n"
        f"Recall    (Sensib.)  : {rec:.4f} ({rec*100:.1f}%)\n"
        f"F1-Score             : {f1:.4f} ({f1*100:.1f}%)\n"
        f"ROC AUC              : {roc_auc_final:.4f}\n"
        "====================================================\n"
    )
    
    with open(OUT_DIR / "metrics_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)
        
    print(report_txt)
    print(f"[✓] Validação real concluída. Gráficos em alta resolução na pasta: {OUT_DIR}")

if __name__ == "__main__":
    run_real_validation()