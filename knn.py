
import tensorflow as tf
import torch
import json
from transformers import  BertForTokenClassification,BertTokenizer, AlbertForTokenClassification,AlbertTokenizer
from transformers import pipeline
albert_model_path='./CZERT-A-ner-CNEC-cased'
bert_model_path='CZERT-B-ner-CNEC-cased'

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
print(outs[0])
print(outs[1])
#model,tokenizer = Open_Bert_Model()
#model,tokenizer = Open_Bert_Model(path="./t1")
#text = "Toto je běžná věta, která používa pojmenované entity, jako Afrika, Roman a Ignác, tiež číslo 025436"
#Use_Print_On_Single_Sentence(model,tokenizer,text)
#Save_Model(model,tokenizer,name="t1")



