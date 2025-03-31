
import tensorflow as tf
import torch
import json
from transformers import  BertForTokenClassification,BertTokenizer, AlbertForTokenClassification,AlbertTokenizer
from transformers import pipeline
import pandas as pd
albert_model_path='./CZERT-A-ner-CNEC-cased'
bert_model_path='CZERT-B-ner-CNEC-cased'

#Toto sa da zmenit v subore config !!! (aj nacitat z neho)
id_to_token = {
        "0": "O",
        "1": "I-T",
        "2": "I-P",
        "3": "I-O",
        "4": "I-M",
        "5": "I-I",
        "6": "I-G",
        "7": "I-A",
        "8": "B-T",
        "9": "B-P",
        "10": "B-O",
        "11": "B-M",
        "12": "B-I",
        "13": "B-G",
        "14": "B-A"
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
    "O":	0
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
    tokenizer = BertTokenizer.from_pretrained(path,vocab_file=path+"/vocab.txt")
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
def Rewrite_To_Bert_Token_Ids(outputs):
    datapieces = []
    id = 0
    for output in outputs:
        tokens = []
        labels = []
        for field in output.tokens:
            #special,type,payload
            tks = field.payload.split()
            for token in tks:
                if token == '':
                    continue
                else:
                    if not field.special:
                        tokens.append(token)
                        labels.append(Token_to_Id("O"))
                    else:
                        tokens.append(token)
                        new_label = Token_to_Token(field.position,field.type)
                        labels.append(Token_to_Id(new_label))

        datapieces.append([tokens,labels])
    return datapieces


#LEEEARNING this is a mess todo
#https://huggingface.co/docs/transformers/en/tasks/token_classification#token-classification
def tokenize_and_align_labels(input,tokenizer):
    tokenized_inputs = tokenizer(input["Token"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(input[f"Label"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["Label"] = labels
    return tokenized_inputs


#Use model
def Use_Print_On_Single_Sentence(model,tokenizer,text):
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    # Perform NER
    entities = ner_pipeline(text)
    # Display results
    for entity in entities:
        print(f"Entity: {entity['word']}, Label: {entity['entity']}, Confidence: {entity['score']}")

'''
# Example: If you want to continue training or fine-tuning, you can use the following:
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# model.train()
# optimizer.zero_grad()
# loss = outputs.loss
# loss.backward()
# optimizer.step()

'''
outs = Parse_CEC_XML_FILE()
datapieces = Rewrite_To_Bert_Token_Ids(outs)
print(datapieces[0])
model,tokenizer = Open_Bert_Model()
df = pd.DataFrame(datapieces, columns=['Token', 'Label'])
print(df.head())
# Save the data into a tab-separated file like in WNUT format
df.to_csv('wnut_like_dataset.txt', sep='\t', index=False, header=False)
tokenized_inputs = tokenize_and_align_labels(datapieces,tokenizer)
print(tokenized_inputs[0])

#model,tokenizer = Open_Bert_Model(path="./t1")
#text = "Býva na adrese Rybná 30 New York."
#Use_Print_On_Single_Sentence(model,tokenizer,text)
#Save_Model(model,tokenizer,name="t1")



