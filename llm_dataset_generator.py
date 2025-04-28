import pandas as pd
import numpy as np 
import re
import os
project1_json = "./project-42-at-2025-03-27-13-14-727622cd.json"
project2_json = "./project-60-at-2025-03-27-13-20-fac55b6a.json"

project1_path = "./HistoryNer/NER"
project2_path = "./HistoryNer/NER_02"
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
df_edit = df[['data']]
print(df.columns)

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

print(df_edit['annotations'].head())

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

#Create new paths
df_edit['edit_file_path'] = df_edit['data'].apply(lambda x: edit_prefix+re.sub(f"^{re.escape(prefix)}", "", x))

#Daump it to files
def save_to_file(file_path,text):
    with open(file_path, "w+") as f:
        f.write(text)

df_edit.apply(lambda row: save_to_file(row['edit_file_path'], row['annotated_text']), axis=1)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df_edit.head(5))
df_edit = df_edit[df_edit['edit_file_path']=="./EHistoryNer/NER/1/000010_1907_uuid-ae853649-d4ee-4b07-af1b-f94bec0f36bf__r001.txt"] #Kde je anotovany len jazyk
print(df_edit.head(1))
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.max_colwidth')

