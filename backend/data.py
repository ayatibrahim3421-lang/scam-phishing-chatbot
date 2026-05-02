import json
import os
from huggingface_hub import hf_hub_download

os.makedirs("backend/data", exist_ok=True)

file_path = hf_hub_download(
    repo_id="ealvaradob/phishing-dataset",
    filename="texts.json",
    repo_type="dataset"
)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

converted = []

for item in data[:2000]:
    text = item["text"]
    label = item["label"]

    converted.append({
        "text": text,
        "label": label,
        "risk": "Phishing" if label == 1 else "Safe",
        "advice": "لا تضغط على الروابط ولا تدخل معلوماتك." if label == 1 else "يبدو آمن لكن تحقق دائماً من المصدر."
    })

with open("backend/data/phishing_knowledge.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print("✅ Done: backend/data/phishing_knowledge.json")