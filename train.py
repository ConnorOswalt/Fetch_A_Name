import json
from collections import defaultdict, Counter
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Load common names from external file
names_file = Path("common_dog_names.txt")
COMMON_DOG_NAMES = set()
if names_file.exists():
    with open(names_file, encoding="utf-8") as f:
        COMMON_DOG_NAMES = {line.strip().title() for line in f if line.strip() and not line.startswith("#")}
else:
    print("⚠️ common_dog_names.txt not found — using minimal fallback")
    COMMON_DOG_NAMES = {"Archie", "Luna", "Bella", "Max"}

def extract_name_candidates(text: str):
    text = text.replace('\n', ' ').strip()
    candidates = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
    # Boost known popular names
    lower = text.lower()
    for name in COMMON_DOG_NAMES:
        if name.lower() in lower:
            candidates.append(name)
    # Clean
    cleaned = {c.strip("'\",.!?").title() for c in candidates if len(c) >= 3}
    return cleaned

# ====================== MAIN ======================
posts_dir = Path("data/posts")
data = []
name_counter = Counter()

for post_folder in tqdm(list(posts_dir.iterdir()), desc="Processing post folders"):
    if not post_folder.is_dir():
        continue

    # Find the JSON
    json_files = list(post_folder.glob("*.json"))
    if not json_files:
        continue
    json_path = json_files[0]  # ← adjust if your JSON has a fixed name

    with open(json_path, encoding="utf-8") as f:
        post = json.load(f)

    # --- Flair bonus ---
    flair = post.get("flair", "")
    flair_name = None
    m = re.search(r'\[.*?(?:name|called|my)?\s*([A-Z][a-zA-Z]{2,})\b.*?\]', flair, re.IGNORECASE)
    if m:
        flair_name = m.group(1).title()

    # --- Vote from comments ---
    name_votes = defaultdict(int)
    if flair_name:
        name_votes[flair_name] += 3

    for comment in post.get("comments", []):
        body = comment.get("body", "")
        score = comment.get("score", 0)
        if score <= 0:
            continue
        for name in extract_name_candidates(body):
            name_votes[name] += score

    if not name_votes:
        continue

    best_name = max(name_votes, key=name_votes.get)
    total_votes = sum(name_votes.values())

    if total_votes >= 5 and name_votes[best_name] >= 2:  # reasonable crowd agreement
        # Add EVERY image in this folder with the same label
        for img_path in post_folder.glob("*.[jJ][pP][gG]"):
            data.append({"image": str(img_path.absolute()), "name": best_name})
            name_counter[best_name] += 1

# Filter rare names
min_freq = 5
valid_names = {n for n, c in name_counter.items() if c >= min_freq}
filtered_data = [row for row in data if row["name"] in valid_names]

df = pd.DataFrame(filtered_data)
df.to_csv("labels.csv", index=False)

print(f"✅ Created {len(df)} labeled images with {len(valid_names)} names")
print("Top names:", name_counter.most_common(10))