import json
import os
import re
import math
import random
from nltk.corpus import wordnet
from copy import deepcopy
from collections import defaultdict

# CONFIG
ORIGINAL_JSON = "project-42-at-2025-03-27-13-14-727622cd.json"
TEXT_ROOT = "./HistoryNer"
OUTPUT_TEXT_DIR = "./augmented_data/texts"
OUTPUT_JSON_PATH = "./augmented_data/augmented_project.json"
TARGET_COUNT = 1500

os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

# These are your class counts
label_counts = {
    'ins': 2057, 'loc_c': 13230, 'amb': 822, 'tim': 4196, 'per': 13371,
    'loc_s': 673, 'misc': 196, 'loc_n': 688, 'obj_a': 1646, 'obj_p': 385,
    'groups': 638, 'ide': 251, 'med': 410, 'evt': 79
}

augmentation_plan = {
    label: math.ceil((TARGET_COUNT - count) / count)
    for label, count in label_counts.items() if count < TARGET_COUNT
}

def load_text_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def synonym_augment(text):
    words = text.split()
    new_words = []
    for word in words:
        if random.random() > 0.9:  # 10% chance to try synonym
            syns = wordnet.synsets(word)
            if syns:
                lemmas = [l.name().replace('_', ' ') for l in syns[0].lemmas()]
                synonyms = list(set(lemmas) - {word})
                if synonyms:
                    word = random.choice(synonyms)
        new_words.append(word)
    return ' '.join(new_words)

def to_xml_formatting(text, annotations):
    begins = []
    ends = []
    for annot in annotations:
        begins.append((annot['start'], annot['labels'][0]))
        ends.append(annot['end'])
    ends.sort()
    begins = sorted(begins, key=lambda x: x[0])
    while ends or begins:
        next_end = ends[-1] if ends else -1
        next_begin = begins[-1][0] if begins else -1
        if next_end > next_begin:
            text = text[:next_end] + "</ne>" + text[next_end:]
            ends.pop()
        else:
            insert = f"<ne type=\"{begins[-1][1]}\">"
            text = text[:next_begin] + insert + text[next_begin:]
            begins.pop()
    return text

def get_labels(results):
    return [r for r in results if r.get('type') == 'labels']

# Load JSON
with open(ORIGINAL_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

augmented_json = []
aug_count = 0

for task in data:
    annotations = task.get("annotations", [])
    for annotation in annotations:
        results = annotation.get("result", [])
        label_annots = get_labels(results)
        if not label_annots:
            continue

        classes = [ann['value']['labels'][0] for ann in label_annots]
        rare_classes = [c for c in classes if c in augmentation_plan]
        if not rare_classes:
            continue

        # Load corresponding .txt file
        file_url = task['data']['text']
        file_path = TEXT_ROOT + re.sub(r'^.*historical_ner', '', file_url)
        file_content = load_text_file(file_path)
        if not file_content:
            continue

        annotated_text = to_xml_formatting(file_content, [r['value'] for r in label_annots])
        multiplier = max([augmentation_plan[c] for c in rare_classes])

        for i in range(multiplier):
            aug_text = synonym_augment(annotated_text)
            output_txt_path = os.path.join(OUTPUT_TEXT_DIR, f"aug_{aug_count}.txt")
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(aug_text)

            # Add new JSON entry
            new_task = deepcopy(task)
            new_task['data']['text'] = output_txt_path
            augmented_json.append(new_task)
            aug_count += 1

print(f"Generated {aug_count} augmented samples.")

# Save new JSON file
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(augmented_json, f, indent=2, ensure_ascii=False)

print(f"Saved augmented JSON to {OUTPUT_JSON_PATH}")
