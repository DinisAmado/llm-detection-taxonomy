import re
import time
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

client = InferenceClient(model=model_id, token=HF_TOKEN)

def classify_text(text):
    messages = [
        {
            "role": "system", 
            "content": (
                "És um analista de sentimentos bilingue. Classifica o texto como:\n"
                "EMOTIONAL: Se contiver palavrões, insultos, choro, desabafos pessoais, "
                "ironia agressiva ou expressões de sentimentos fortes.\n"
                "OTHER: Se for um relato de rotina, factos neutros, política ou lógica.\n"
                "Responde APENAS com a palavra 'EMOTIONAL' ou 'OTHER'."
            )
        },
        {"role": "user", "content": f"Classifica este texto: '{text}'"}
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=5,
            temperature=0.1
        )
        
        prediction = response.choices[0].message.content.strip().upper()
        
        if "EMOTIONAL" in prediction: return "EMOTIONAL"
        if "OTHER" in prediction: return "OTHER"
        return "OTHER"
        
    except Exception as e:
        print(f"\n[DEBUG] Erro na API: {e}")
        return "ERRO_API"

def run_test():
    stats = {"EMOTIONAL": 0, "OTHER": 0, "ERRO_API": 0}
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
                result = classify_text(clean_text)
                stats[result] = stats.get(result, 0) + 1
                print(f"{result:<15} | {clean_text[:75]}...")
                
                time.sleep(1.0)

        print("\n" + "="*30)
        print(f"RESUMO DO TESTE:")
        print(f"Emocionais: {stats['EMOTIONAL']}")
        print(f"Outros: {stats['OTHER']}")
        print("="*30)

    except FileNotFoundError:
        print("Erro: O arquivo 'examples.txt' não foi encontrado.")

if __name__ == "__main__":
    run_test()