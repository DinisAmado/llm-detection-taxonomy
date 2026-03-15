import re
import time
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
model_id = "meta-llama/Llama-Guard-3-8B"

client = InferenceClient(model=model_id, token=HF_TOKEN)

def classify_threat(text):
    messages = [
        {
            "role": "system", 
            "content": (
                "És um especialista em segurança e moderação. Classifica o texto como:\n"
                "THREAT: Se contiver ameaças de violência física, planos de ataque, "
                "intenção de causar danos a propriedades ou pessoas.\n"
                "OTHER: Se for um desabafo, crítica ou facto sem ameaça física.\n"
                "Responde APENAS com a palavra 'THREAT' ou 'OTHER'."
            )
        },
        {"role": "user", "content": f"Verifica se este texto é uma ameaça: '{text}'"}
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=5,
            temperature=0.1
        )
        prediction = response.choices[0].message.content.strip().upper()
        
        if "THREAT" in prediction: return "THREAT"
        return "OTHER"
        
    except Exception as e:
        print(f"\n[DEBUG] Erro na API: {e}")
        return "ERRO_API"

def run_test():
    stats = {"THREAT": 0, "OTHER": 0, "ERRO_API": 0}
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
                result = classify_threat(clean_text)
                stats[result] = stats.get(result, 0) + 1
                print(f"{result:<15} | {clean_text[:75]}...")
                
                time.sleep(1.0)

        print("\n" + "="*30)
        print(f"RESUMO DO TESTE (THREAT):")
        print(f"Ameaças: {stats.get('THREAT', 0)}")
        print(f"Outros: {stats.get('OTHER', 0)}")
        print("="*30)

    except FileNotFoundError:
        print("Erro: O ficheiro 'examples.txt' não foi encontrado.")

if __name__ == "__main__":
    run_test()