import re
import time
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
model_id = "meta-llama/Llama-Guard-3-8B"

client = InferenceClient(model=model_id, token=HF_TOKEN)

def classify_hate(text):    
    messages = [
        {
            "role": "system", 
            "content": (
                "És um especialista em moderação de discurso de ódio. Classifica o texto como:\n"
                "HATE: Se contiver insultos graves, discriminação, ataques a grupos minoritários, "
                "desumanização ou linguagem abusiva direcionada.\n"
                "OTHER: Se for um desabafo comum, crítica rudes mas não odiosa, ou factos.\n"
                "Responde APENAS com a palavra 'HATE' ou 'OTHER'."
            )
        },
        {"role": "user", "content": f"Analise este comentário: '{text}'"}
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=5,
            temperature=0.1
        )
        prediction = response.choices[0].message.content.strip().upper()
        
        if "HATE" in prediction: return "HATE"
        return "OTHER"
        
    except Exception as e:
        print(f"\n[DEBUG] Erro na API: {e}")
        return "ERRO_API"

def run_test():
    stats = {"HATE": 0, "OTHER": 0, "ERRO_API": 0}
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
                result = classify_hate(clean_text)
                stats[result] = stats.get(result, 0) + 1
                print(f"{result:<15} | {clean_text[:75]}...")
                
                time.sleep(1.0)

        print("\n" + "="*30)
        print(f"RESUMO DO TESTE (HATE):")
        print(f"Hate/Ódio: {stats.get('HATE', 0)}")
        print(f"Outros: {stats.get('OTHER', 0)}")
        print("="*30)

    except FileNotFoundError:
        print("Erro: O ficheiro 'examples.txt' não foi encontrado.")

if __name__ == "__main__":
    run_test()