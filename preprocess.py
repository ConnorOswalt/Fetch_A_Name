import json
from collections import defaultdict, Counter
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Safe spaCy loading
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    print("✅ spaCy loaded")
except Exception:
    print("⚠️ spaCy not available - using regex only")
    nlp = None
    SPACY_AVAILABLE = False


def extract_proposed_dog_names(text: str):
    if not text or len(text.strip()) < 3:
        return set()

    text = text.replace('\n', ' ').strip()
    candidates = set()

    # Direct suggestion patterns
    patterns = [
        r'^(?:[Ii] suggest|[Mm]y vote|[Cc]all him|[Cc]all her|[Nn]ame him|[Nn]ame her|he is a|she is a|this is a|looks like a?)\s+([A-Z][a-zA-Z]{2,})\b',
        r'\b(?:name|call|him|her|this|that)\s+([A-Z][a-zA-Z]{2,})\b',
    ]
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            name = match.group(1).strip("'\",.!?").title()
            if len(name) >= 3:
                candidates.add(name)

    # Capitalized word fallback
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
    for word in capitalized:
        word = word.strip("'\",.!?").title()
        if len(word) >= 3 and word not in {"This", "That", "Call", "Name", "Looks", "Vote", "Suggest", "He", "She", "Him", "Her", "Reddit", "Google"}:
            candidates.add(word)

    # spaCy PERSON (if available)
    if SPACY_AVAILABLE and nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip("'\",.!?").title()
                if len(name) >= 3:
                    candidates.add(name)

    blacklist = {"Reddit", "Google", "The", "A", "An"}
    candidates = {c for c in candidates if c not in blacklist and not c.isupper()}
    return candidates


# ====================== MAIN ======================
posts_dir = Path("data/posts")
data = []
name_counter = Counter()

for post_folder in tqdm(list(posts_dir.iterdir()), desc="Processing posts"):
    if not post_folder.is_dir():
        continue

    json_files = list(post_folder.glob("*.json"))
    if not json_files:
        continue
    json_path = json_files[0]

    try:
        with open(json_path, encoding="utf-8") as f:
            post = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {json_path.name}: {e}")
        continue

    # --- Flair bonus (stronger weight because it's often the chosen name) ---
    flair = post.get("flair")
    flair_name = None
    if flair and isinstance(flair, str):
        m = re.search(r'\[.*?(?:name|called|my)?\s*([A-Z][a-zA-Z]{2,})\b.*?\]', flair, re.IGNORECASE)
        if m:
            flair_name = m.group(1).strip().title()
        else:
            words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', flair)
            if len(words) == 1:
                candidate = words[0].strip().title()
                if len(candidate) >= 3:
                    flair_name = candidate

    # --- Vote counting with likes weighting ---
    name_votes = defaultdict(int)

    # Flair bonus (high weight)
    if flair_name:
        name_votes[flair_name] += 15          # ← strong bonus for OP flair

    # Process every comment
    for comment in post.get("comments", []):
        body = comment.get("body", "")
        score = comment.get("score", 0)

        if score <= 0 or not isinstance(body, str) or len(body.strip()) < 3:
            continue   # ← drop negative / zero score comments completely

        for name in extract_proposed_dog_names(body):
            name_votes[name] += score          # weight by actual likes

    if not name_votes:
        continue

    best_name = max(name_votes, key=name_votes.get)
    total_votes = sum(name_votes.values())
    winning_votes = name_votes[best_name]

    # Only accept if there's decent crowd support
    if total_votes >= 5 and winning_votes >= 2:
        # Assign the best name to EVERY image in this post folder
        for img_path in post_folder.glob("*.[jJ][pP][gG]"):
            if img_path.is_file():
                data.append({"image": str(img_path.absolute()), "name": best_name})
                name_counter[best_name] += 1

# Filter rare names
min_freq = 5
valid_names = {n for n, c in name_counter.items() if c >= min_freq}
filtered_data = [row for row in data if row["name"] in valid_names]

df = pd.DataFrame(filtered_data)
df.to_csv("labels.csv", index=False)

print(f"\n✅ Finished!")
print(f"   Labeled images : {len(df)}")
print(f"   Unique names   : {len(valid_names)}")
print(f"   Top 10 names   : {name_counter.most_common(10)}")