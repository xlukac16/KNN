
import tensorflow as tf
import torch
import json
from transformers import BertForTokenClassification, BertTokenizerFast, AlbertForTokenClassification, AlbertTokenizer, DataCollatorForTokenClassification
from transformers import pipeline, TrainingArguments, Trainer
import pandas as pd
import numpy as np # yay for numpy
from datasets import Dataset, DatasetDict
import evaluate

albert_model_path='./CZERT-A-ner-CNEC-cased'
bert_model_path='CZERT-B-ner-CNEC-cased'

#Toto sa da zmenit v subore config !!! (aj nacitat z neho)
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
token_to_token = {
    "ah":	"A",
    "at":	"A",
    "az":	"A",
    "a":	"A",
    "gh":	"G",
    "gq":	"G",
    "gs":	"G",
    "gu":	"G",
    "g_":	"G",
    "gt":	"G",
    "gr":	"G",
    "gl":	"G",
    "g":	"G",
    "ia":	"I",
    "i_":	"I",
    "if":	"I",
    "ic":	"I",
    "io":	"I",
    "i":	"I",
    "mi":	"M",
    "me":	"M",
    "ms":	"M",
    "mn":	"M",
    "m":	"M",
    "nb":	"I",
    "ni":	"I",
    "ns":	"I",
    "na":	"A",
    "nc":	"A",
    "no":	"A",
    "n_":	"A",
    "n":	"A",
    "oa":	"M",
    "om":	"M",
    "or":	"M",
    "oe":	"M",
    "op":	"M",
    "o_":	"M",
    "o":	"M",
    "pc":	"P",
    "pf":	"P",
    "pp":	"P",
    "p_":	"P",
    "pd":	"P",
    "pm":	"P",
    "ps":	"P",
    "p":	"P",
    "tf":	"T",
    "tm":	"T",
    "td":	"T",
    "th":	"T",
    "ty":	"T",
    "t":	"T"
}

# Load, Save the model

def Open_Albert_Model(path=albert_model_path):
    tokenizer = AlbertTokenizer.from_pretrained(path,vocab_file=path+"/vocab.txt")
    model = AlbertForTokenClassification.from_pretrained(path)
    return model,tokenizer

#Funguje aj na nacitavanie natrenovaneho a ulozeneho modelu
def Open_Bert_Model(path=bert_model_path):
    tokenizer = BertTokenizerFast.from_pretrained(path,vocab_file=path+"/vocab.txt")
    model = BertForTokenClassification.from_pretrained(path)
    return model,tokenizer

def Save_Model(model,tokenizer,name="saved_model"):
    model.save_pretrained("./"+name)
    tokenizer.save_pretrained("./"+name)







#Work training data
import xml.etree.ElementTree as ET
import re
xml_file_path='./CEC2.0/data/xml/named_ent.xml'

class Single_Token():
    special=False
    position='O'
    type=None
    payload=None
    def __init__(self,g_payload,g_type=None):
        if g_type is None:
            self.special = False
        else:
            self.special=True
            self.type = g_type
        self.payload = g_payload
    def __repr__(self):
        if self.special:
            return f"({self.special} {self.position}-{self.type} {self.payload})"
        else:
            return f"({self.special} {self.position} {self.payload})"
    
class Single_Input():
    tokens = []
    def __init__(self,line: str):
        parts = re.findall(r'<ne type="([^"]+)">(.*?)</ne>|([^<]+)', line)
        self.tokens = []
        for part in parts:
            if part[0]:
                self.tokens.append(Single_Token(part[1],part[0]))
            if part[2]:
                if part[2].strip() == '':
                    continue
                self.tokens.append(Single_Token(part[2]))
        lt=False
        for token in self.tokens:
            if token.special:
                if not lt:
                    lt = True
                    token.position='B'
                else:
                    token.position='I'
            else:
                lt=False
    def __repr__(self):
        return f"tokens={self.tokens}"

def Parse_CEC_XML_FILE(file_path=xml_file_path):
    with open(file_path, 'r') as file:
        xml_content = file.readlines()
    pattern = r'</?doc>'
    outputs = []
    for line in xml_content:
        match = re.match(pattern,line)
        if not match:
            outputs.append(Single_Input(line))
    return outputs

def Token_to_Id(token_in):
    return token_to_id.get(token_in, 0)
def Id_to_Token(id_in):
    return id_to_token.get(id_in,"O")

'''
    Vsetko co neviem zaradit inde je misc
'''
def Token_to_Token(pos_in='B',token_in='M'):
    token_out = pos_in+'-'+token_to_token.get(token_in,"M")
    return token_out

#IT just works it just works it just works ....
#There is no evaluate set, split will beeeeeee 85-15 between train test, becausseee I said so
'''
def Rewrite_To_Bert_Token_Ids(outputs,tokenizer):
    train_set = []
    test_set = []
    split=len(outputs) * 0.8
    train_id = 0
    test_id = 0
    train_payloads = []
    test_payloads = []
    train_labels = []
    test_labels = []
    for output in outputs:
        tokens = []
        labels = []
        string_payload=""
        for field in output.tokens:
            #special,type,payload
            string_payload += field.payload
            tks = tokenizer(field.payload,truncation=True,padding=True,max_length=128, add_special_tokens=True)
            tks.word_ids(batch_index=0)
            decoded_text = tokenizer.convert_ids_to_tokens(tks["input_ids"])
            for token_id,decoded_token in zip(tks["input_ids"],decoded_text):
                if (decoded_token[0] == '[') or (decoded_token[0] == '#' and decoded_token[1] == '#'):
                    tokens.append(token_id)
                    labels.append(Token_to_Id("err"))
                elif not field.special:
                    tokens.append(token_id)
                    labels.append(Token_to_Id("O"))
                else:
                    tokens.append(token_id)
                    new_label = Token_to_Token(field.position,field.type)
                    labels.append(Token_to_Id(new_label))

        ones_array = np.ones(len(tokens))

        if train_id < split:
            train_payloads.append(string_payload)
            train_labels.append(labels)
            train_set.append({
                'id':torch.tensor(train_id),
                'input_ids':torch.tensor(tokens),
                'ner_tags':torch.tensor(labels),
                'attention_mask':torch.tensor(ones_array)
                })
            train_id+=1
        else:
            test_payloads.append(string_payload)
            test_labels.append(labels)
            test_set.append({
                'id':torch.tensor(test_id),
                'input_ids':torch.tensor(tokens),
                'ner_tags':torch.tensor(labels),
                'attention_mask':torch.tensor(ones_array)
                })
            test_id+=1
    print(train_payloads[0])
    print(train_labels[0])
    test_tks = tokenizer(test_payloads,truncation=True, add_special_tokens=True)
    train_tks= tokenizer(train_payloads,truncation=True, add_special_tokens=True)
    test_tks["labels"] = test_labels
    train_tks["labels"] = train_labels
        
    return train_set,test_set,train_tks,test_tks
'''



def sort_to_datasets_and_connect(data,tokenizer):
    train_id = 0
    test_id = 0
    train_set = []
    test_set = []
    split=len(data) * 0.8
    for output in data:
        tokens = []
        labels = []
        string_payload=""
        for field in output.tokens:
            #special,type,payload
            string_payload += field.payload
            tks = tokenizer(field.payload,truncation=True,padding=True,max_length=128, add_special_tokens=True)
            decoded_text = tokenizer.convert_ids_to_tokens(tks["input_ids"])
            for token_id,decoded_token in zip(tks["input_ids"],decoded_text):
                if (decoded_token[0] == '[') or (decoded_token[0] == '#' and decoded_token[1] == '#'):
                    continue
                elif not field.special:
                    tokens.append(token_id)
                    labels.append(Token_to_Id("O"))
                else:
                    tokens.append(token_id)
                    new_label = Token_to_Token(field.position,field.type)
                    labels.append(Token_to_Id(new_label))
        if train_id < split:
            train_set.append({
                'id':train_id,
                'tokens':string_payload,
                'ner_tags':labels,
                })
            train_id+=1
        else:
            test_set.append({
                'id':test_id,
                'tokens':string_payload,
                'ner_tags':labels
                })
            test_id+=1
    return train_set,test_set


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=False)
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
    return tokenized_inputs
#LEEEARNING this is a mess
#https://huggingface.co/docs/transformers/en/tasks/token_classification#token-classification


#Use model
def Use_Print_On_Single_Sentence(model,tokenizer,text):
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    # Perform NER
    entities = ner_pipeline(text)
    # Display results
    for entity in entities:
        print(f"Entity: {entity['word']}, Label: {entity['entity']}, Confidence: {entity['score']}")


#this was never tested
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






outs = Parse_CEC_XML_FILE()
model,tokenizer = Open_Bert_Model()
train_list,test_list = sort_to_datasets_and_connect(outs,tokenizer)
train_dataset = Dataset.from_list(train_list)
test_dataset = Dataset.from_list(test_list)
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})
train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized = test_dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")

training_args = TrainingArguments(
    output_dir="my_awesome_wnut_model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
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

# Save the data into a tab-separated file like in WNUT format

#tokenized_inputs = tokenize_and_align_labels(datapieces,tokenizer)
#print(tokenized_inputs[0])

#model,tokenizer = Open_Bert_Model(path="./t1")
#text = "Býva na adrese Rybná 30 New York."
#Use_Print_On_Single_Sentence(model,tokenizer,text)
#Save_Model(model,tokenizer,name="t1")



