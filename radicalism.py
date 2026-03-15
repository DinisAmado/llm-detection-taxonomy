import re
import time
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

client = InferenceClient(model=model_id, token=HF_TOKEN)

def classify_radicalism(text):
    messages = [
        {
            "role": "system", 
            "content": (
                "És um analista de segurança. Classifica o texto como:\n"
                "RADICALISM: Se contiver apelos à derrubada de instituições, "
                "rejeição total de valores democráticos ou extremismo político.\n"
                "OTHER: Se for uma opinião política comum, desabafo ou facto.\n"
                "Responde APENAS com a palavra 'RADICALISM' ou 'OTHER'."
            )
        },
        {"role": "user", "content": f"Analise: '{text}'"}
    ]
    
    try:
        response_obj = client.chat_completion(
            messages=messages,
            max_tokens=400,
            temperature=0.1
        )
        
        full_content = response_obj.choices[0].message.content
        
        answer = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip().upper()
        
        if "RADICALISM" in answer: return "RADICALISM"
        return "OTHER"
        
    except Exception as e:
        print(f"\n[DEBUG] Erro: {e}")
        return "ERRO_API"

def run_test():
    stats = {"RADICALISM": 0, "OTHER": 0, "ERRO_API": 0}
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
                result = classify_radicalism(clean_text)
                stats[result] = stats.get(result, 0) + 1
                print(f"{result:<15} | {clean_text[:75]}...")
                
                time.sleep(1.0)

        print("\n" + "="*30)
        print(f"RESUMO (RADICALISMO):")
        print(f"Radicalismo: {stats.get('RADICALISM', 0)}")
        print(f"Outros: {stats.get('OTHER', 0)}")
        print("="*30)

    except FileNotFoundError:
        print("Erro: Ficheiro 'examples.txt' não encontrado.")

if __name__ == "__main__":
    run_test()