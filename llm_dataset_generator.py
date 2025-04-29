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

def debug_print_df_head(df_edit):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    #print(df_edit.columns)
    print(df_edit.head(5))
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

def remove_the_path(string):
    return re.sub("^.*\/","",string)
def get_anotator(string):
    parts = re.split('\/',string)
    return parts[len(parts)-2]

#Split dataset by anotator
df_edit['file_name'] = df_edit['file_path'].apply(lambda x: remove_the_path(x))
df_edit['anotator'] = df_edit['file_path'].apply(lambda x: get_anotator(x))
df_edit['count'] = df_edit.groupby('file_name')['file_name'].transform('count')
df_edit['annot_c'] = df_edit.groupby('anotator')['anotator'].transform('count')

df_anot1 = df_edit[df_edit['anotator']=='1']
df_anot2 = df_edit[df_edit['anotator']=='2']
df_anot3 = df_edit[df_edit['anotator']=='3']
df_anot4 = df_edit[df_edit['anotator']=='4']
df_anot5 = df_edit[df_edit['anotator']=='5']

def split_dataset(df_edit:pd.DataFrame):
    duplicates = df_edit[df_edit.duplicated('file_name', keep=False)]
    singles = df_edit[~df_edit.duplicated('file_name', keep=False)]
    duplicates_len = duplicates.shape[0]
    if duplicates_len>int(len(df_edit)*(1 - 0.85)):
        return singles,duplicates
    move = int(len(df_edit)*(1 - 0.85) - duplicates_len)
    part1 = df_edit.iloc[:move]
    part2 = df_edit.iloc[move:]
    test = pd.concat([duplicates, part1], ignore_index=True)
    return test,part2

df_anot1_test,df_anot1_train = split_dataset(df_anot1)
df_anot2_test,df_anot2_train = split_dataset(df_anot2)
df_anot3_test,df_anot3_train = split_dataset(df_anot3)
df_anot4_test,df_anot4_train = split_dataset(df_anot4)
df_anot5_test,df_anot5_train = split_dataset(df_anot5)

df_train_f = pd.concat([df_anot1_train, df_anot2_train], ignore_index=True)
df_train_f = pd.concat([df_train_f, df_anot3_train], ignore_index=True)
df_train_f = pd.concat([df_train_f, df_anot4_train], ignore_index=True)
df_train_f = pd.concat([df_train_f, df_anot5_train], ignore_index=True)



from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments,DataCollatorForLanguageModeling

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add padding token
tokenizer.pad_token = tokenizer.eos_token
def tokenize(text_in,text_out):
    full_text = f"Input: {text_in}\nOutput: {text_out}"
    return tokenizer(full_text, truncation=True, padding='max_length', max_length=512)

print(df_train_f.columns)

df_train_f['tokenized']=df_train_f.apply(lambda row: tokenize(row['file_contents'], row['annotated_text']), axis=1)
df_anot1_test['tokenized']=df_anot1_test.apply(lambda row: tokenize(row['file_contents'], row['annotated_text']), axis=1)

model.resize_token_embeddings(len(tokenizer))
training_args = TrainingArguments(
    output_dir="./ner-gpt",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    evaluation_strategy="epoch",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # GPT-2 is not masked
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=df_train_f['tokenized'],
    eval_dataset=df_anot1_test['tokenized'],
    tokenizer=tokenizer,
    data_collator=data_collator
)
trainer.train()
prompt = "Extract entities from: Barack Obama was born in Hawaii.\nEntities:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

