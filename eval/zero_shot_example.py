from transformers import pipeline
from huggingface_hub import login
import os

HF_TOKEN = os.environ.get("HF_TOKEN")


file = open("examples.txt", "r", encoding="utf-8")
examples = file.readlines()
file.close()
login(token =  HF_TOKEN)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = ["emotional", "extremist", "hate", "radicalism", "threat", "violated","sentimental"]

d_result={}

for text in examples:
    result = classifier(text.strip(), candidate_labels, multi_label=True)
    d_result.update({ text.strip(): { "label": result['labels'][0], "acurracy": round(result['scores'][0], 4)}})



print(d_result)