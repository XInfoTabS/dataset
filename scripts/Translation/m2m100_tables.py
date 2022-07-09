import time, os, json, math, string
from collections import Counter,OrderedDict
from typing import Any, Optional, Dict
from pprint import pprint
import argparse
from tqdm import trange
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np

import torch
import torchtext as tt

import spacy
ner = spacy.load("en_core_web_sm")


def config(parser):
    parser.add_argument('--tables_path',  default='../../data/en/tables', type=str)
    parser.add_argument('--context_file_path',  default='../../utilities/additional_data/table_categories.tsv', type=str)
    parser.add_argument('--translation_lang',  default='fr', type=str)
    parser.add_argument('--original_lang',  default='en', type=str)
    parser.add_argument('--save_path',  default='../../data/fr/tables', type=str)
  
    return parser


"""## Creating utility functions"""

def translate(text : str, src_code : str = "en", trg_code : str = "en", model = None, tokenizer = None,  device: str = 'cpu') -> str :
  tokenizer.src_lang = src_code
  encoded = tokenizer(text, return_tensors="pt").to(device)
  generated_tokens = model.generate(
      **encoded,
      forced_bos_token_id = tokenizer.get_lang_id(trg_code)
  )
  res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
  return res[0]

# util to load tables
def json_to_dict(path : str) -> dict:
    with open(path, 'r') as file:
        result_dict = dict(json.loads(file.read()))
    return result_dict
    
# util to save tables
def dict_to_json(obj : Dict, path: str) -> None:
  with open(path, 'w') as file:
    file.write(json.dumps(obj, indent=4))

# ----- MAIN FUNCTION -----

def table_translate_M2M100 (tables_path: str, context_file_path:str, og_lang_code:str, tr_lang_code:str, save_path:str):

  #loading the device for the model
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  #loading the tokenizer and the model
  tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
  model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B").to(device)

  #loading the file names for all the tables
  tables = os.listdir(tables_path)
  print(f"Total number of tables to translate: {len(tables)}")

  #loading the save_directory for the file
  if not os.path.isdir(save_path):
    os.mkdir(save_path)
    print("Directory created!")
  else:
    print("Directory already exists!")

  table_prefix = tr_lang_code + '_'
  tables_already_translated = [table_name.replace(table_prefix, "") for table_name in os.listdir(save_path)]
  tables_left = [a for a in infotabs_tables if a not in tables_already_translated]
  print(f"Tables left to translate: {len(tables_left)}")

  cat_table = pd.read_csv(context_file_path, delimiter='\t', index_col=0)

  for table_name in tqdm(tables_left):
    # getting the table
    table_path = tables_path + '/' + table_name
    table = json_to_dict(table_path)

    # getting table category
    table_code = table_name.replace(".json", "")
    print(table_code)

    try:
      category = cat_table.loc[table_code].category
    except:
      category = 'None'
    
    # print(category)

    # converting table --> context sentences
    table_texts = []
    for key, values in table.items():
      context = category + " | " + key + " | "

      ner_text = ner(values[0])
      ner_flag = all([(a.tag_ == 'NNP' or a.tag_ == 'NNPS' or a.tag_ == 'NNS' or a.tag_ == 'NN')  
                                for a in ner_text if str(a) not in string.punctuation])

      for value in values:
        if ner_flag:                           # NER checking for Proper Nouns
          # print(value)
          text = context + '"' + value + '"'
        else: 
          text = context + value 
        
        table_texts.append(text)

    # pprint(table_texts)

    # translating the table_texts
    
    tr_table_texts = []
    tr_punk_1 = '////'
    tr_punk_2 = '<PAD>'

    err_1 = 0
    err_2 = 0

    for text in tqdm(table_texts):
      # print(text)
      
      text_1 = text.replace("|", tr_punk_1)
      # print(text_1)
      tr_text = translate(text_1, og_lang_code, tr_lang_code, model, tokenizer, device)
      # print(tr_text, '\n')

      if len(tr_text.split(tr_punk_1)) == 3 :
        tr_text = tr_text.replace(tr_punk_1, '|')
        tr_table_texts.append(tr_text)
      else:
        err_1 += 1

        text_2 = text.replace("|", tr_punk_2)
        tr_text = translate(text_2, og_lang_code, tr_lang_code,  model, tokenizer, device)

        if len(tr_text.split(tr_punk_2)) == 3 :
          tr_text = tr_text.replace(tr_punk_2, '|')
          tr_table_texts.append(tr_text)
        else:
          err_2 += 1

          cat, key, val = text.split(' | ')

          tr_cat = translate(cat, og_lang_code, tr_lang_code, model, tokenizer, device)
          tr_key = translate(key, og_lang_code, tr_lang_code, model, tokenizer, device)
          tr_val = translate(val, og_lang_code, tr_lang_code,  model, tokenizer, device)
          
          tr_text = tr_cat + " | " + tr_key + " | " + tr_val
          tr_table_texts.append(tr_text)

    print(f"{100 * err_1/len(table_texts)} {100 * err_2 / len(table_texts)}")

    # recreating the table from the texts
    tr_table = {}
    for text in tr_table_texts:
      try:
        cat, key, value = [x.strip() for x in text.split('|')]
      except:
        assert len([x.strip() for x in text.split('|')]) != 3, print(f"len: {len([x.strip() for x in text.split('|')])} ") 

      value = value.replace('"','')   # Removing quotes from NER values

      if key in tr_table.keys():
        tr_table[key].append(value)
      else:
        tr_table[key] = []
        tr_table[key].append(value)
      
    # pprint(tr_table)

    # saving the table in the save file
    table_prefix = tr_lang_code                 # Set a table prefix to differentiate tables
    table_save_path = save_path + '/' + table_prefix + '_' + table_name 
    dict_to_json(tr_table, table_save_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser = config(parser)
  args = vars(parser.parse_args())

  table_translate_M2M100(args['tables_path'], args['context_file_path'], args['original_lang'], args['translation_lang'], args['save_path'])