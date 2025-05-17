import re
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss
import os
import random
from collections import Counter
from bs4 import BeautifulSoup

project1_json = r"D:\Programming - Big data/project-42-at-2025-03-27-13-14-727622cd.json"
project2_json = r"D:\Programming - Big data/project-60-at-2025-03-27-13-20-fac55b6a.json"

project1_path = r"D:\Programming - Big data/HistoryNer/NER"
project2_path = r"D:\Programming - Big data/HistoryNer/NER_02"
prefix = "https://label-studio.semant.cz/data/local-files/?d=historical_ner"
my_prefix = "./HistoryNer"
edit_prefix = "./EHistoryNer"


def load_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read()  # You can adjust what data you want to read from the file
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    else:
        return None

df = pd.read_json(project1_json)
df_edit = df[['data']].copy()
#print(df.columns)

#Opem files to ['file_contents']
df_edit['data'] = df_edit['data'].apply(lambda x: x['text'])
df_edit['file_path'] = df_edit['data'].apply(lambda x: my_prefix+re.sub(f"^{re.escape(prefix)}", "", x))
df_edit['file_contents'] = df_edit['file_path'].apply(load_file)

#Remove weird annotations for now
df_edit['annotations'] = df[['annotations']]
df_edit['annotation_count'] = df_edit['annotations'].apply(lambda x: len(x))
df_edit = df_edit[df_edit['annotation_count']==1]
df_edit['annotations'] = df_edit['annotations'].apply(lambda x: x[0])
df_edit['annotations'] = df_edit['annotations'].apply(lambda x: x['result'])

#filter podla jazyku
df_edit['language'] =  df_edit['annotations'].apply(lambda x: [annotation['value']['choices'][0] for annotation in x if annotation['type'] == 'choices'])
df_edit['language_count'] = df_edit['language'].apply(lambda x: len(x))
df_edit = df_edit[df_edit['language_count']==1] #Niektore tam nemali
df_edit['language'] = df_edit['language'].apply(lambda x: x[0])
df_edit = df_edit[df_edit['language']=='czech'] #Niektore tam nemali
df_edit = df_edit.drop('language_count', axis=1)

df_edit['annotations'] =  df_edit['annotations'].apply(lambda x: [annotation['value'] for annotation in x if annotation['type'] == 'labels'])
df_edit['annotations_count'] = df_edit['annotations'].apply(lambda x: len(x))
df_edit = df_edit[df_edit['annotations_count']>0] #Kde je anotovany len jazyk
df_edit = df_edit.drop('annotations_count', axis=1)

#print(df_edit['annotations'].head())

#edit text to xml_formatting
def to_xml_formatting(text,annotations):
    begins = []
    ends = []
    for annot in annotations:
        begins.append((annot['start'],annot['labels'][0]))
        ends.append(annot['end'])
    ends.sort()
    begins = sorted(begins, key=lambda x: x[0])
    while(len(ends)!=0 or len(begins)!=0):
        next_end = ends[len(ends)-1] if len(ends)-1 >= 0 else -1
        next_begin = begins[len(begins)-1][0] if len(begins)-1 >= 0 else -1
        if next_end > next_begin:
            insert = "</ne>"
            text = text[:next_end] + insert + text[next_end:]
            ends.pop()
        else:
            insert = f"<ne type=\"{begins[len(begins)-1][1]}>\""
            text = text[:next_begin] + insert + text[next_begin:]
            begins.pop()
    return text

df_edit['annotated_text'] = df_edit.apply(lambda row: to_xml_formatting(row['file_contents'], row['annotations']), axis=1)
#pd.set_option('display.max_colwidth', None)
#print(df_edit['annotated_text'].head(3))

# Dummy LLM output
def corrupt_some_entities(text):
    if "<ne type=location>Brna</ne>" in text and random.random() < 0.3:
        return text.replace("Brna", "Prahy")
    return text

df_edit['llm_output'] = df_edit['annotated_text'].apply(corrupt_some_entities)

def noisy_prediction(text):
    if random.random() < 0.3:
        text = text.replace("á", "a").replace("ě", "e")  # simulate LLM weirdness
    return text

df_edit['llm_output'] = df_edit['llm_output'].apply(noisy_prediction)
#df_edit['llm_output'] = df_edit['annotated_text'].copy()


def extract_plain_text(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def extract_all_entities(text):
    pattern = r"<ne type\s*=\s*([^>]+)>(.*?)</ne>"
    return re.findall(pattern, text, re.DOTALL)

def compute_preservation_score(reference, prediction):
    ref_clean = extract_plain_text(reference).split()
    pred_clean = extract_plain_text(prediction).split()
    matched = sum(r == p for r, p in zip(ref_clean, pred_clean))
    return matched / max(len(ref_clean), len(pred_clean)) if max(len(ref_clean), len(pred_clean)) > 0 else 0.0

def compute_ner_metrics_all(reference, prediction):
    ref_entities = extract_all_entities(reference)
    pred_entities = extract_all_entities(prediction)

    ref_counter = Counter(ref_entities)
    pred_counter = Counter(pred_entities)

    y_true_vals = []
    y_pred_probs = []

    all_entities = set(ref_counter.keys()).union(pred_counter.keys())

    for entity in all_entities:
        ref_count = ref_counter[entity]
        pred_count = pred_counter[entity]

        min_count = min(ref_count, pred_count)
        extra_ref = ref_count - min_count
        extra_pred = pred_count - min_count

        # True positives (correctly predicted)
        y_true_vals.extend([1] * min_count)
        y_pred_probs.extend([random.uniform(0.85, 1.0)] * min_count)

        # False negatives (missed by model)
        y_true_vals.extend([1] * extra_ref)
        y_pred_probs.extend([random.uniform(0.0, 0.3)] * extra_ref)

        # False positives (model predicted but not in gold)
        y_true_vals.extend([0] * extra_pred)
        y_pred_probs.extend([random.uniform(0.7, 1.0)] * extra_pred)

    if not y_true_vals:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "loss": 0.0,
        }

    y_pred_hard = [1 if p > 0.5 else 0 for p in y_pred_probs]

    return {
        "f1": f1_score(y_true_vals, y_pred_hard),
        "precision": precision_score(y_true_vals, y_pred_hard),
        "recall": recall_score(y_true_vals, y_pred_hard),
        "accuracy": accuracy_score(y_true_vals, y_pred_hard),
        "loss": log_loss(y_true_vals, y_pred_probs, labels=[0, 1]),
    }


def evaluate_llm_output(reference: str, prediction: str):
    preservation = compute_preservation_score(reference, prediction)
    ner_metrics = compute_ner_metrics_all(reference, prediction)
    
    return {
        "preservation_score": preservation,
        "ner_f1_score": ner_metrics["f1"],
        "ner_precision": ner_metrics["precision"],
        "ner_recall": ner_metrics["recall"],
        "ner_accuracy": ner_metrics["accuracy"],
        "eval_loss": ner_metrics["loss"],
    }


def evaluate_batch(references, predictions):
    assert len(references) == len(predictions), "The lists must have the same length"
    
    results = []
    for ref, pred in zip(references, predictions):
        result = evaluate_llm_output(ref, pred)
        results.append(result)
    
    def avg(metric):
        return sum(r[metric] for r in results) / len(results)

    return {
        "average_preservation_score": avg("preservation_score"),
        "average_ner_f1_score": avg("ner_f1_score"),
        "average_ner_precision": avg("ner_precision"),
        "average_ner_recall": avg("ner_recall"),
        "average_ner_accuracy": avg("ner_accuracy"),
        "average_eval_loss": avg("eval_loss"),
        "detailed_scores": results
    }


references = df_edit['annotated_text'].tolist()
predictions = df_edit['llm_output'].tolist()

result = evaluate_batch(references, predictions)

print("Preservation score:", result["average_preservation_score"])
print("F1 skóre (eval_f1)):", result["average_ner_f1_score"])
print("Presnosť (eval_precision):", result["average_ner_precision"])
print("Úplnosť (eval_recall):", result["average_ner_recall"])
print("Presnosť tokenov (eval_accuracy):", result["average_ner_accuracy"])
print("Strata (eval_loss):", result["average_eval_loss"])
#print("Detailed scores:", result['detailed_scores'])

# Export do CSV
#df = pd.DataFrame(result["detailed_scores"])
#df.to_csv("llm_evaluation_results.csv", index=False)