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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def config(parser):
    parser.add_argument('--hypothesis_path',  default='../../data/en/en_train.tsv', type=str)
    parser.add_argument('--tables_path',  default='../../data/en/tables/', type=str)
    parser.add_argument('--translation_lang',  default='fr', type=str)
    parser.add_argument('--original_lang',  default='en', type=str)
    parser.add_argument('--save_path',  default='../../data/fr/tables', type=str)
  
    return parser

def translate(text : str, model = None, tokenizer = None,  device: str = 'cpu') -> str :
  encoded = tokenizer(text, truncation=True, return_tensors="pt").to(device)
  generated_tokens = model.generate(**encoded)
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


def hypothesis_translate_mBART (hypo_path:str, tables_path:str, og_lang_code:str, tr_lang_code:str, save_path: str):
    if hypo_path.split('.')[-1] == 'tsv':
        hypo_df = pd.read_csv(hypo_path, delimiter='\t', index_col=0)
    elif hypo_path.split('.')[-1] == 'csv':
        hypo_df = pd.read_csv(hypo_path, index_col=0)
    else:
        assert(False, "File type not supported, please provide .csv or .tsv")

    hypothesis_texts = list(hypo_df['hypothesis'])
    table_ids = list(hypo_df['table_id'])

    tr_hypothesis_texts = []

    #loading the device for the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #loading the tokenizer and the model
    tr_model_code = "Helsinki-NLP/opus-mt-" + og_lang_code + "-" + tr_lang_code

    tokenizer = AutoTokenizer.from_pretrained(tr_model_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(tr_model_code).to(device)

    for i in trange(len(hypothesis_texts)):
        hypo = hypothesis_texts[i]
        table_id = table_ids[i]

        # getting the table
        table_path = tables_path + '/' + table_id + '.json'
        table = json_to_dict(table_path)

        #getting and quoting table title in the text
        title = table['title'][0]
        hypo = hypo.replace(title, '"' + title + '"')
        
        tr_hypo = translate(hypo, model, tokenizer, device)
        tr_hypo = tr_hypo.replace('"', '')

        # print(tr_hypo)

        tr_hypothesis_texts.append(tr_hypo)


    #saving the file in the position
    tr_hypo_train_df = pd.DataFrame.from_dict({
        'annotater_id': list(hypo_df['annotater_id']),
        'table_id' : list(hypo_df['table_id']),
        'hypothesis' : tr_hypothesis_texts,
        'label' : list(hypo_df['label'])
        })

    file_prefix = tr_lang_code
    train_save_path = save_path + '/' + file_prefix + '_hypothesis_train.csv'

    tr_hypo_train_df.to_csv(train_save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = config(parser)
    args = vars(parser.parse_args())
    

    hypothesis_translate_mBART(args['hypothesis_path'],args['tables_path'], args['original_lang'], args['translation_lang'], args['save_path'])