import re
import time
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
model_id = "meta-llama/Llama-3.2-3B-Instruct"

client = InferenceClient(model=model_id, token=HF_TOKEN)

def classify_violated(text):
    """Classifica se o texto expressa um sentimento de violação (privacidade, direitos, limites)."""
    
    messages = [
        {
            "role": "system", 
            "content": (
                "És um analista de sentimentos. Classifica o texto como:\n"
                "VIOLATED: Se o texto expressar invasão de espaço pessoal, quebra de privacidade, "
                "abuso de confiança, manipulação, perseguição ou violação de direitos.\n"
                "OTHER: Se for um desabafo emocional comum, factos ou opiniões neutras.\n"
                "Responde APENAS com a palavra 'VIOLATED' ou 'OTHER'."
            )
        },
        {"role": "user", "content": f"Analise este conteúdo: '{text}'"}
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=5,
            temperature=0.1
        )
        prediction = response.choices[0].message.content.strip().upper()
        
        if "VIOLATED" in prediction: return "VIOLATED"
        return "OTHER"
        
    except Exception as e:
        print(f"\n[DEBUG] Erro na API: {e}")
        return "ERRO_API"

def run_test():
    stats = {"VIOLATED": 0, "OTHER": 0, "ERRO_API": 0}
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
                result = classify_violated(clean_text)
                stats[result] = stats.get(result, 0) + 1
                print(f"{result:<15} | {clean_text[:75]}...")
                
                time.sleep(0.8)

        print("\n" + "="*30)
        print(f"RESUMO (VIOLATED):")
        print(f"Violados: {stats.get('VIOLATED', 0)}")
        print(f"Outros: {stats.get('OTHER', 0)}")
        print("="*30)

    except FileNotFoundError:
        print("Erro: O ficheiro 'examples.txt' não foi encontrado.")

if __name__ == "__main__":
    run_test()