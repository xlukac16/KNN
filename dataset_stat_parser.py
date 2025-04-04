import json
from collections import Counter
import zipfile
import os
import tempfile
import matplotlib.pyplot as plt
import re

# JSON header parser (for historical)
file_1 = r"{{}}\project-42-at-2025-03-27-13-14-727622cd.json"
file_2 = r"{{}}\project-60-at-2025-03-27-13-20-fac55b6a.json"
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

# raw data parser (for historical) (this one might take a while)
zip_path = r"{{}}\Historical_NER_raw_text_data.zip"

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
file_path = r"{{}}\named_ent.txt"

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