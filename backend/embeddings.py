import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "phishing_knowledge.json"
OUT_PATH = BASE_DIR / "data" / "phishing_embeddings.json"

LIMIT = 2000
BATCH_SIZE = 100

def prepare_text(item):
    text = f"""
Type: {item.get("risk", "")}
Message: {item.get("text", "")}
Advice: {item.get("advice", "")}
""".strip()

    # قص النصوص الطويلة جدًا
    return text[:6000]

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

data = data[:LIMIT]

output = []

for i in range(0, len(data), BATCH_SIZE):
    batch = data[i:i + BATCH_SIZE]
    texts = [prepare_text(item) for item in batch]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    for item, emb in zip(batch, response.data):
        output.append({
            "item": item,
            "embedding": emb.embedding
        })

    print(f"✅ Embedded {i + len(batch)} / {len(data)}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print("✅ Done: data/phishing_embeddings.json")