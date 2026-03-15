import re
import time
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"

client = InferenceClient(model=model_id, token=HF_TOKEN)

def classify_sentimental(text):
    try:
        results = client.text_classification(text)
        
        best_result = max(results, key=lambda x: x['score'])
        label = best_result['label'].lower()

        if label in ['positive', 'negative']:
            return "SENTIMENTAL"
        return "OTHER"
        
    except Exception as e:
        print(f"\n[DEBUG] Erro na API: {e}")
        return "ERRO_API"

def run_test():
    stats = {"SENTIMENTAL": 0, "OTHER": 0, "ERRO_API": 0}
    try:
        with open("examples.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        print(f"\n{'RESULTADO':<15} | {'TEXTO'}")
        print("-" * 85)

        for line in lines:
            line = line.strip()
            if not line or len(line) < 5: continue
                
            clean_text = re.sub(r'\\', '', line).strip()
            
            if clean_text:
                result = classify_sentimental(clean_text)
                stats[result] = stats.get(result, 0) + 1
                print(f"{result:<15} | {clean_text[:75]}...")
                
                time.sleep(0.5)

        print("\n" + "="*30)
        print(f"RESUMO (SENTIMENTAL):")
        print(f"Sentimentais: {stats.get('SENTIMENTAL', 0)}")
        print(f"Outros: {stats.get('OTHER', 0)}")
        print("="*30)

    except FileNotFoundError:
        print("Erro: Ficheiro 'examples.txt' não encontrado.")

if __name__ == "__main__":
    run_test()