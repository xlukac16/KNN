import json
from collections import Counter
import zipfile
import os
import tempfile
import matplotlib.pyplot as plt
import re


# JSON header parser (for historical)
file_1 = r"D:\Programming - Big data\project-42-at-2025-03-27-13-14-727622cd.json"
file_2 = r"D:\Programming - Big data\project-60-at-2025-03-27-13-20-fac55b6a.json"
with open(file_1, 'r', encoding='utf-8') as f:
    data = json.load(f)
label_counter = Counter()

for task in data:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "labels":
                labels = result.get("value", {}).get("labels", [])
                label_counter.update(labels)

with open(file_2, 'r', encoding='utf-8') as f:
    data = json.load(f)

for task in data:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "labels":
                labels = result.get("value", {}).get("labels", [])
                label_counter.update(labels)

print("Total number of labels:", sum(label_counter.values()))
print("Unique label types and counts:")
for label, count in label_counter.items():
    print(f"  {label}: {count}")
labels = list(label_counter.keys())
counts = list(label_counter.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color='skyblue')
plt.xlabel('NER Label')
plt.ylabel('Count')
plt.title('NER Label Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# raw data parser (for historical) (this one might take a while)
zip_path = r"D:\Programming - Big data\\Historical_NER_raw_text_data.zip"

lengths = []
word_counts = []

with tempfile.TemporaryDirectory() as temp_dir:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        lengths.append(len(text))
                        word_counts.append(len(text.split()))
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.violinplot(lengths)
plt.title("Dĺžky súborov v datasete (počet znakov)")
plt.xlabel("Znaky")

plt.subplot(1, 2, 2)
plt.violinplot(word_counts)
plt.title("Dĺžky súborov v datasete (počet slov)")
plt.xlabel("Slová")

plt.tight_layout()
plt.savefig("historical_word_count_plots.png", dpi=300)
plt.show()


# parser for named_ent.txt (for cnec)
file_path = r"D:\Programming - Big data\\named_ent.txt"

lengths = []
word_counts = []

tag_pattern = re.compile(r'<[^<>]*>')

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        clean_line = tag_pattern.sub('', line)
        clean_line = re.sub(r'\s+', ' ', clean_line).strip()

        lengths.append(len(clean_line))
        word_counts.append(len(clean_line.split()))

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.violinplot(lengths)
plt.title("Dĺžky súborov v datasete (počet znakov)")
plt.xlabel("Znaky")

plt.subplot(1, 2, 2)
plt.violinplot(word_counts)
plt.title("Dĺžky súborov v datasete (počet slov)")
plt.xlabel("Slová")

plt.tight_layout()
plt.savefig("cnec_word_count_plots.png", dpi=300)
plt.show()


# === Configuration ===
json_path = "D:\Programming - Big data\project-42-at-2025-03-27-13-14-727622cd.json"
text_file_prefix = "./HistoryNer"  # Local prefix for linked text files
url_prefix = "https://label-studio.semant.cz/data/local-files/?d=historical_ner"

# === Functions ===

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def local_file_path(web_path):
    return text_file_prefix + re.sub(f"^{re.escape(url_prefix)}", "", web_path)

def count_labels_from_annotations(data):
    label_counter = Counter()
    for item in data:
        annotations = item.get("annotations", [])
        if not annotations:
            continue
        # Use only the first annotation (you can adapt this to handle multiples)
        results = annotations[0].get("result", [])
        for result in results:
            if result["type"] == "labels":
                labels = result["value"]["labels"]
                for label in labels:
                    label_counter[label] += 1
    return label_counter

# === Execution ===

data = load_json(json_path)
label_counts = count_labels_from_annotations(data)

print("NER Label Distribution:")
for label, count in label_counts.items():
    print(f"{label}: {count}")
