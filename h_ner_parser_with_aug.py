import pandas as pd
import re
import os
from transformers import  BertForTokenClassification,BertTokenizerFast, DataCollatorForTokenClassification
import evaluate
from transformers import pipeline, TrainingArguments, Trainer
from datasets import Dataset,DatasetDict
import numpy as np # yay for numpy

project1_json = "./project-42-at-2025-03-27-13-14-727622cd.json"
project2_json = "./project-60-at-2025-03-27-13-20-fac55b6a.json"
#project3_json = "./augmented_project.json"

project1_path = "./HistoryNer/NER"
project2_path = "./HistoryNer/NER_02"
#project3_path = "./HistoryNer/texts"
prefix = "https://label-studio.semant.cz/data/local-files/?d=historical_ner"
my_prefix = "./HistoryNer"

#TODO toto bude treba skonzultovat
token_to_token = {
    "loc_c" : "G",
    "amb"   : "M",
    "tim"   : "T",
    "per"   : "P",
    "loc_s" : "G",
    "misc"  : "M",
    "ins"   : "I",
    "obj_a" : "M",
    "loc_n" : "G",
    "obj_p" : "M",
    "groups": "I",
    "ide"   : "M",
    "med"   : "M",
    "evt"   : "M",
    "O"     : "O"
}

token_to_id = {
    "B-A":	14,
    "B-G":	13,
    "B-I":	12,
    "B-M":	11,
    "B-O":	10,
    "B-P":	9,
    "B-T":	8,
    "I-A":	7,
    "I-G":	6,
    "I-I":	5,
    "I-M":	4,
    "I-O":	3,
    "I-P":	2,
    "I-T":	1,
    "O":	0,
    "err": -100
}

id_to_token = {
        0: "O",
        1: "I-T",
        2: "I-P",
        3: "I-O",
        4: "I-M",
        5: "I-I",
        6: "I-G",
        7: "I-A",
        8: "B-T",
        9: "B-P",
        10: "B-O",
        11: "B-M",
        12: "B-I",
        13: "B-G",
        14: "B-A",
        -100: "err"
    }

# Keep only labels with at least 1500 occurrences
MIN_COUNT = 1500
label_counts = {
    'ins': 2057,
    'loc_c': 13230,
    'amb': 822,
    'tim': 4196,
    'per': 13371,
    'loc_s': 673,
    'misc': 196,
    'loc_n': 688,
    'obj_a': 1646,
    'obj_p': 385,
    'groups': 638,
    'ide': 251,
    'med': 410,
    'evt': 79,
}

# Set of labels we want to keep
VALID_LABELS = {label for label, count in label_counts.items() if count >= MIN_COUNT}

def load_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                return file.read()  # You can adjust what data you want to read from the file
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    else:
        return None

#dame 0.8 train, 0.1 eval,0.1 test -> 0 train , 1 eval, 2 test
# Load JSON file into a Pandas DataFrame
#TODO - fix multiple annotations in df_edit['annotations'](I just dropped it)
#TODO - rework train set to accomodate for multiple annotations for same file (it's ignored right now)
df = pd.read_json(project1_json)
df_edit = df[['data']]
print(df.columns)

#Opem files to ['file_contents']
df_edit['data'] = df_edit['data'].apply(lambda x: x['text'])
df_edit['file_path'] = df_edit['data'].apply(lambda x: my_prefix+re.sub(f"^{re.escape(prefix)}", "", x))
df_edit['file_contents'] = df_edit['file_path'].apply(load_file)

#Split ['set'] to train, eval, test
train_split = 0.8 * df_edit.shape[0]
eval_split = train_split + 0.1 * df_edit.shape[0]
df_edit['set'] = 2
df_edit['set'] = df_edit.loc[:eval_split, 'set'] = 1
df_edit['set'] = df_edit.loc[:train_split, 'set'] = 0

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
#3951 poloziek zde

#odtraneneie nasobnych anotacii for now
def multi_removal(dict_list):
    new_dict_list = dict_list
    inter_list = []
    for dict in dict_list:
        inter_list.append((dict['start'],dict['end']))
    inter_list.sort(key=lambda x: x[0])
    while True:
        idx = -1
        for i in range(1,len(inter_list)):
            if inter_list[i][0] <= inter_list[i - 1][1]:
                idx = i if (inter_list[i][1] - inter_list[i][0]) > (inter_list[i-1][1] - inter_list[i-1][0]) else i-1

        if (idx == -1):
            break
        else:
            inter_list.pop(idx)
            new_dict_list.pop(idx)
    return new_dict_list


df_edit['annotations'] = df_edit['annotations'].apply(lambda x: multi_removal(x))



#Split text na bloky
def text_split(text, annot_list):
    annot_list = [a for a in annot_list if a['labels'][0] in VALID_LABELS]
    annot_list.sort(key=lambda x: x['start'])
    begin_idx = 0
    ret_annot_list = []
    for annot in annot_list:
        part1 = text[begin_idx:annot['start']]
        part2 = text[annot['start']:annot['end']]
        begin_idx = annot['end']
        if part1.strip() != "":
            ret_annot_list.append((part1, "O"))
        if part2.strip() != "":
            ret_annot_list.append((part2, annot['labels'][0]))
    remaining = text[begin_idx:]
    if remaining.strip() != "":
        ret_annot_list.append((remaining, "O"))
    return ret_annot_list

df_edit['annotations'] = df_edit.apply(lambda x: text_split(x['file_contents'],x['annotations']), axis=1)

#Generate tokens
bert_model_path='CZERT-B-ner-CNEC-cased'
def Open_Bert_Model(path=bert_model_path):
    tokenizer = BertTokenizerFast.from_pretrained(path,vocab_file=path+"/vocab.txt")
    model = BertForTokenClassification.from_pretrained(path)
    # comment out if using just the CPU
    model.to("cuda")
    return model,tokenizer
model,tokenizer = Open_Bert_Model()

def token_splitter(annotation_blocks,tokenizer):
    tokens = []
    for at in annotation_blocks:
        tks = tokenizer(at[0],truncation=True)
        for token in tks['input_ids']:
            tokens.append(at[1])
    return tokens


df_edit['tokens'] = df_edit['annotations'].apply(lambda x: token_splitter(x,tokenizer))
print(df_edit.iloc[0]['tokens'])

def token_switcher(tokens):
    ntks = []
    for tk in tokens:
        ntks.append(token_to_token[tk])
    return ntks

df_edit['tokens'] = df_edit['tokens'].apply(lambda x: token_switcher(x))
print(df_edit.iloc[0]['tokens'])

def add_positions(tokens):
    last_token = 'O'
    tks = []
    for token in tokens:
        if token == 'O':
            tks.append(token)
        elif token != last_token:
            token="B-"+token
            tks.append(token)
        else:
            token="I-"+token
            tks.append(token)
        last_token=token
    return tks

df_edit['tokens'] = df_edit['tokens'].apply(lambda x: add_positions(x))
print(df_edit.iloc[0]['tokens'])

def token_to_idf(tokens):
    ntks = []
    for tk in tokens:
        ntks.append(token_to_id[tk])
    return ntks

df_edit['tokens'] = df_edit['tokens'].apply(lambda x: token_to_idf(x))
print(df_edit.iloc[0]['tokens'])

#unique_values = []
#df_edit['tokens'].apply(lambda x: [unique_values.append(item) for item in x if item not in unique_values])
#print(unique_values)
#['O', ['loc_c'], ['amb'], ['tim'], ['per'], ['loc_s'], ['misc'], ['ins'], ['loc_n'], ['obj_a'], ['obj_p'], ['groups'], ['ide'], ['med'], ['evt']]

df_edit = df_edit.reset_index()
df_edit = df_edit.drop('data', axis=1)
df_edit = df_edit.drop('file_path', axis=1)
df_edit['ner_tags'] = df_edit['tokens']
df_edit = df_edit.drop('tokens', axis=1)
df_edit = df_edit.drop('language', axis=1)
df_edit = df_edit.drop('annotations', axis=1)
df_edit = df_edit.drop('annotation_count', axis=1)
df_edit['tokens'] = df_edit['file_contents']
df_edit = df_edit.drop('file_contents', axis=1)
print(df_edit.columns)
train_df = df_edit[df_edit['set'] == 0]
eval_df = df_edit[df_edit['set'] == 1]
test_df = df_edit[df_edit['set'] == 2]
train_df = train_df.drop('set', axis=1)
eval_df = eval_df.drop('set', axis=1)
test_df = test_df.drop('set', axis=1)

train_dict = train_df.to_dict(orient='records')
evaldict = eval_df.to_dict(orient='records')
test_dict = train_df.to_dict(orient='records')
train_dataset = Dataset.from_list(train_dict)
test_dataset = Dataset.from_list(test_dict)
print(train_dict[0])

print("ENDER")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=False, padding=True, max_length=512)
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                try:
                    label_ids.append(label[word_idx])
                except:
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    while True:
        if len(tokenized_inputs["labels"]) >= len(tokenized_inputs['input_ids']):
            break
        else:
            tokenized_inputs["labels"].append(-100)
    return tokenized_inputs

def compute_metrics(p):
    predictions,labels = p
    predictions = np.argmax(predictions, axis=-1)
    true_predictions = [
        [id_to_token[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_token[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized = test_dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")

training_args = TrainingArguments(
    output_dir="my_awesome_wnut_model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("traintime")

trainer.train()
print("done")

eval_results = trainer.evaluate()
print(eval_results)





