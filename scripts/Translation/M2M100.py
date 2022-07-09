# basic files
import time, os, json, os, math
from collections import Counter,OrderedDict
from typing import Any, Optional
from pprint import pprint

import pandas as pd
import numpy as np

import fuzzywuzzy as fw
from fuzzywuzzy import fuzz

import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

import torch
import torchtext as tt
from torchtext.data.metrics import bleu_score

from tqdm.notebook import tqdm, trange

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# loading the infotabs english files 
infotabs_tables_path = "../../data/en/tables"
infotabs_tables = os.listdir(infotabs_tables_path)
print(f"Number of tables in infotabs: {len(infotabs_tables)}")

subset_path = "../input/xinfotabs/subsets/subset_300.json"
with open(subset_path, "r") as file:
    subset = json.loads(file.read())
print(f"Size of the subset: {100*subset['size']:.2f}% ")

# loading the category table
cat_table_path = "../input/xinfotabs/table_categories.tsv"
cat_table = pd.read_csv(cat_table_path, delimiter='\t', index_col=0)
cat_table.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T07:50:51.112851Z","iopub.execute_input":"2021-12-21T07:50:51.113164Z","iopub.status.idle":"2021-12-21T07:50:51.12064Z","shell.execute_reply.started":"2021-12-21T07:50:51.113135Z","shell.execute_reply":"2021-12-21T07:50:51.119403Z"}}
# example of taking cateogory from table
cat_table.loc['T1008'].category

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T07:50:51.452359Z","iopub.execute_input":"2021-12-21T07:50:51.452675Z","iopub.status.idle":"2021-12-21T07:50:51.457229Z","shell.execute_reply.started":"2021-12-21T07:50:51.452647Z","shell.execute_reply":"2021-12-21T07:50:51.456117Z"}}
# util to load tables
def json_to_dict(path : str) -> dict:
    with open(path, 'r') as file:
        result_dict = dict(json.loads(file.read()))
    return result_dict

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T07:50:55.205347Z","iopub.execute_input":"2021-12-21T07:50:55.205665Z","iopub.status.idle":"2021-12-21T07:50:57.868803Z","shell.execute_reply.started":"2021-12-21T07:50:55.205635Z","shell.execute_reply":"2021-12-21T07:50:57.867946Z"}}
# creating category | key | value sentences
subset_texts = []
for table_code in subset['subset']:
    # loading the table
    table_path = infotabs_tables_path + '/' + table_code + '.json'
    table = json_to_dict(table_path)
    
    # getting the category
    try:
        category = cat_table.loc[table_code].category
    except:
        category = 'None'
        
    # getting all the values for each key  
    for key in table.keys():
        subset_texts += [category + ' | ' + key + ' | ' + value for value in table[key]]
        
subset_texts = list(set(subset_texts))
print(f"Length of the subset inputs: {len(subset_texts)}")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T07:50:57.871847Z","iopub.execute_input":"2021-12-21T07:50:57.87211Z","iopub.status.idle":"2021-12-21T07:54:38.507302Z","shell.execute_reply.started":"2021-12-21T07:50:57.872083Z","shell.execute_reply":"2021-12-21T07:54:38.505358Z"}}
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B").to(device)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T07:54:38.509667Z","iopub.execute_input":"2021-12-21T07:54:38.51003Z","iopub.status.idle":"2021-12-21T07:54:38.517566Z","shell.execute_reply.started":"2021-12-21T07:54:38.509993Z","shell.execute_reply":"2021-12-21T07:54:38.516669Z"}}
def translate(text : str, src_code : str = "en", trg_code : str = "en", model = None, tokenizer = None,  device: str = 'cpu') -> str :
  tokenizer.src_lang = src_code
  encoded = tokenizer(text, return_tensors="pt").to(device)
  generated_tokens = model.generate(
      **encoded,
      forced_bos_token_id = tokenizer.get_lang_id(trg_code)
  )
  res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
  return res[0]

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T07:54:38.519039Z","iopub.execute_input":"2021-12-21T07:54:38.519394Z","iopub.status.idle":"2021-12-21T07:54:39.671338Z","shell.execute_reply.started":"2021-12-21T07:54:38.519357Z","shell.execute_reply":"2021-12-21T07:54:39.670369Z"}}
# test of translation
translate('Hello.', src_code='en',  trg_code='zh', model = model, tokenizer = tokenizer, device=device)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T07:54:39.672671Z","iopub.execute_input":"2021-12-21T07:54:39.673175Z","iopub.status.idle":"2021-12-21T07:54:39.67738Z","shell.execute_reply.started":"2021-12-21T07:54:39.673137Z","shell.execute_reply":"2021-12-21T07:54:39.676359Z"}}
# declare all the langauge codes
og_lang_code = "en"
tr_lang_code = "zh"

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T07:58:06.614808Z","iopub.execute_input":"2021-12-21T07:58:06.615213Z","iopub.status.idle":"2021-12-21T10:26:44.233338Z","shell.execute_reply.started":"2021-12-21T07:58:06.615178Z","shell.execute_reply":"2021-12-21T10:26:44.23229Z"}}
# translating all the texts to another language

# since the error of direct splitting is too much, we need to employ another method
# If it can be split directly, it would be split directly
# Else, we use '////' and '<PAD>' tokens because they seem to work the best out of tried tokens
# If nothing works, we'll split the input, translate everything and then combine


tr_subset_texts = []


punc_1 = '////'
punc_2 = '<PAD>'

err_1 = []   # can't be split directly into 3 parts
err_2 = []   # can't be split with punc_1 into 3 parts
err_3 = []   # can't be split with punc_2 into 3 parts


for i in trange(len(subset_texts)):
    text = subset_texts[i]
    
    
    tr_text_dir = translate(text, og_lang_code, tr_lang_code, model, tokenizer, device)
    tr_text_dir_list = tr_text_dir.split('|')
    
    if len(tr_text_dir_list) == 3:
        tr_subset_text.append(tr_text_dir)
    else:
        err_1.append(i)
        tr_text_punc_1 = translate(text.replace("|", punc_1), og_lang_code, tr_lang_code, model, tokenizer, device)
        tr_text_punc_1_list = tr_text_punc_1.split(punc_1)
        if len(tr_text_punc_1_list) == 3:
            tr_subset_texts.append(tr_text_punc_1.replace(punc_1, "|"))
        else: 
            err_2.append(i)
            tr_text_punc_2 = translate(text.replace("|", punc_2), og_lang_code, tr_lang_code, model, tokenizer, device)
            tr_text_punc_2_list = tr_text_punc_2.split(punc_2)
            if len(tr_text_punc_2_list) == 3:
                tr_subset_texts.append(tr_text_punc_2.replace(punc_2, "|"))
            else:
                err_3.append(i)
                cat = text.split(" | ")[0].strip()
                key = text.split(" | ")[1].strip()
                value = text.split(" | ")[2].strip()
                
                tr_cat = translate(cat, og_lang_code, tr_lang_code, model, tokenizer, device)
                tr_key = translate(key, og_lang_code, tr_lang_code, model, tokenizer, device)
                tr_value = translate(value, og_lang_code, tr_lang_code,model, tokenizer, device)
                
                tr_text = tr_cat + ' | ' + tr_key + ' | ' + tr_value
                                       
                tr_subset_texts.append(tr_text)
        
    
print(f"Length of translated texts: {len(tr_subset_texts)}")
print(f"Errors: { len(err_1) } { len(err_2) } { len(err_3) }")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T10:26:44.237563Z","iopub.execute_input":"2021-12-21T10:26:44.237953Z","iopub.status.idle":"2021-12-21T10:26:44.298727Z","shell.execute_reply.started":"2021-12-21T10:26:44.237914Z","shell.execute_reply":"2021-12-21T10:26:44.297711Z"}}
temp_df = pd.DataFrame.from_dict({"subset_texts":subset_texts,
                                  "tr_subset_texts":tr_subset_texts
                                 })
temp_df.to_csv("./ZH_temp_context_attached_M2M100.csv")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T10:26:44.303446Z","iopub.execute_input":"2021-12-21T10:26:44.305493Z","iopub.status.idle":"2021-12-21T10:26:44.310682Z","shell.execute_reply.started":"2021-12-21T10:26:44.305449Z","shell.execute_reply":"2021-12-21T10:26:44.309896Z"}}
# temp_df = pd.read_csv("../input/xinfotabs/temp/RU_temp_context_attached_M2M100.csv",index_col=0)
# subset_texts = list(temp_df['subset_texts'])
# tr_subset_texts = list(temp_df['tr_subset_texts'])

# print(f"{ len(subset_texts) }  { len(tr_subset_texts) }")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T10:26:44.314984Z","iopub.execute_input":"2021-12-21T10:26:44.317707Z","iopub.status.idle":"2021-12-21T10:26:44.32838Z","shell.execute_reply.started":"2021-12-21T10:26:44.31766Z","shell.execute_reply":"2021-12-21T10:26:44.327358Z"}}
# # experiments with tokens for attaching context
# ind = 2
# print(subset_texts[ind].replace("|","</s>"))
# print(translate(subset_texts[ind].replace("|","</s>"), og_lang_code, tr_lang_code, model, tokenizer, device))

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2021-12-21T10:26:44.333584Z","iopub.execute_input":"2021-12-21T10:26:44.336054Z","iopub.status.idle":"2021-12-21T10:26:44.342279Z","shell.execute_reply.started":"2021-12-21T10:26:44.336011Z","shell.execute_reply":"2021-12-21T10:26:44.341369Z"}}
# # running experiments on attached context seperation for 100 random samples

# err_1 = 0
# err_2 = 0
# samples = np.random.randint(low = 0, high = len(subset_texts), size = (100, ))
# print(samples)

# punc = '</s>'
# punc_sub = '</s>'
# for i in tqdm(samples):
    
#     tr_text = translate(subset_texts[i].replace("|",punc), og_lang_code, tr_lang_code, model, tokenizer, device)
    
#     try: 
#         tr_text_list = tr_text.split(punc)
        
#         if len(tr_text_list) != 3:
#             assert len(tr_text_list) == 3
        
#         key = tr_text_list[1].strip()
    
#     except:
#         err_1+=1
        
#         tr_text = translate(subset_texts[i].replace("|",punc_sub), og_lang_code, tr_lang_code, model, tokenizer, device)
        
#         try: 
#             tr_text_list = tr_text.split(punc_sub)
        
#             if len(tr_text_list) != 3:
#                 assert len(tr_text_list) == 3
        
#             key = tr_text_list[1].strip()
            
#         except:
#             err_2+=1
        
        
# print(f"{ len(samples) }  { err_1 } { err_2 } ")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T10:26:44.347185Z","iopub.execute_input":"2021-12-21T10:26:44.348563Z","iopub.status.idle":"2021-12-21T10:26:44.377949Z","shell.execute_reply.started":"2021-12-21T10:26:44.34852Z","shell.execute_reply":"2021-12-21T10:26:44.37714Z"}}
# having a look at the translations

for text in tr_subset_texts:
    text_list = text.split(" | ")
    if len(text_list) != 3 :  
        print(f" {text} ")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T10:26:44.381576Z","iopub.execute_input":"2021-12-21T10:26:44.383442Z","iopub.status.idle":"2021-12-21T12:36:19.581601Z","shell.execute_reply.started":"2021-12-21T10:26:44.383397Z","shell.execute_reply":"2021-12-21T12:36:19.580714Z"}}
# back translating all the text to original language
bt_subset_texts = []

# since the error of direct splitting is too much, we need to employ another method
# If it can be split directly, it would be split directly
# Else, we use '////' and '<PAD>' tokens because they seem to work the best out of tried tokens
# If nothing works, we'll split the input, translate everything and then combine



punc_1 = '////'
punc_2 = '<PAD>'

bt_err_1 = []   # can't be split directly into 3 parts
bt_err_2 = []   # can't be split with punc_1 into 3 parts
bt_err_3 = []   # can't be split with punc_2 into 3 parts


for i in trange(len(tr_subset_texts)):
    text = tr_subset_texts[i]
    
    bt_text_dir = translate(text, tr_lang_code, og_lang_code, model, tokenizer, device)
    bt_text_dir_list = bt_text_dir.split('|')
    
    if len(bt_text_dir_list) == 3:
        bt_subset_text.append(bt_text_dir)
    else:
        bt_err_1.append(i)
        bt_text_punc_1 = translate(text.replace("|", punc_1), tr_lang_code, og_lang_code, model, tokenizer, device)
        bt_text_punc_1_list = bt_text_punc_1.split(punc_1)
        if len(bt_text_punc_1_list) == 3:
            bt_subset_texts.append(bt_text_punc_1.replace(punc_1, "|"))
        else: 
            bt_err_2.append(i)
            bt_text_punc_2 = translate(text.replace("|", punc_2), tr_lang_code, og_lang_code, model, tokenizer, device)
            bt_text_punc_2_list = bt_text_punc_2.split(punc_2)
            if len(bt_text_punc_2_list) == 3:
                bt_subset_texts.append(bt_text_punc_2.replace(punc_2, "|"))
            else:
                bt_err_3.append(i)
                
                try: 
                    cat = text.split(" | ")[0].strip()
                except:
                    cat = "none"
                
                try:
                    key = text.split(" | ")[1].strip()
                except:
                    key = "none"
                
                try:
                    value = text.split(" | ")[2].strip()
                except:
                    value = "none"
                
                bt_cat = translate(cat, tr_lang_code, og_lang_code, model, tokenizer, device)
                bt_key = translate(key, tr_lang_code, og_lang_code, model, tokenizer, device)
                bt_value = translate(value, tr_lang_code, og_lang_code,model, tokenizer, device)
                
                bt_text = bt_cat + ' | ' + bt_key + ' | ' + bt_value
                                       
                bt_subset_texts.append(bt_text)
        
    
print(f"Length of translated texts: {len(bt_subset_texts)}")
print(f"Errors: { len(bt_err_1) } { len(bt_err_2) } { len(bt_err_3) }")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.584392Z","iopub.execute_input":"2021-12-21T12:36:19.584962Z","iopub.status.idle":"2021-12-21T12:36:19.59217Z","shell.execute_reply.started":"2021-12-21T12:36:19.584922Z","shell.execute_reply":"2021-12-21T12:36:19.591402Z"}}
bt_dict = {"bt":bt_subset_texts}
with open("./bt.json", "w") as file:
    file.write(json.dumps(bt_dict,))

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.593995Z","iopub.execute_input":"2021-12-21T12:36:19.594347Z","iopub.status.idle":"2021-12-21T12:36:19.645673Z","shell.execute_reply.started":"2021-12-21T12:36:19.594312Z","shell.execute_reply":"2021-12-21T12:36:19.644787Z"}}
bt_texts_df = pd.DataFrame.from_dict({"original_texts":subset_texts,
                                      "translated_texts": tr_subset_texts,
                                      "backtranslated_texts":bt_subset_texts
                                     })

bt_texts_df.to_csv("./ZH_context_texts_backtranslation_error_analysis_M2M100_300_v1.csv")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.646891Z","iopub.execute_input":"2021-12-21T12:36:19.647398Z","iopub.status.idle":"2021-12-21T12:36:19.655448Z","shell.execute_reply.started":"2021-12-21T12:36:19.647359Z","shell.execute_reply":"2021-12-21T12:36:19.654439Z"}}
subset_keys = [text.split(' | ')[1] for text in subset_texts]
print(f"Length of subset keys: {len(subset_keys)} and example: {subset_keys[1]}")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.656919Z","iopub.execute_input":"2021-12-21T12:36:19.65763Z","iopub.status.idle":"2021-12-21T12:36:19.669048Z","shell.execute_reply.started":"2021-12-21T12:36:19.657552Z","shell.execute_reply":"2021-12-21T12:36:19.668206Z"}}
tr_subset_keys = [text.split('|')[1].strip() for text in tr_subset_texts]
print(f"Length of subset keys: {len(tr_subset_keys)} and example: {tr_subset_keys[1]}")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.671453Z","iopub.execute_input":"2021-12-21T12:36:19.671755Z","iopub.status.idle":"2021-12-21T12:36:19.682681Z","shell.execute_reply.started":"2021-12-21T12:36:19.67172Z","shell.execute_reply":"2021-12-21T12:36:19.681874Z"}}
bt_subset_keys = []
for i, text in enumerate(bt_subset_texts):
    try:
        bt_key =  text.split('|')[1].strip()
    except:
        print(i)
        bt_key = "None"
    bt_subset_keys.append(bt_key)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.684074Z","iopub.execute_input":"2021-12-21T12:36:19.684652Z","iopub.status.idle":"2021-12-21T12:36:19.712493Z","shell.execute_reply.started":"2021-12-21T12:36:19.684616Z","shell.execute_reply":"2021-12-21T12:36:19.711647Z"}}
# saving the context keys
bt_keys_df = pd.DataFrame.from_dict({"original_keys":subset_keys,
                                      "translated_keys": tr_subset_keys,
                                      "backtranslated_keys":bt_subset_keys
                                     })

bt_keys_df.to_csv("./ZH_context_keys_backtranslation_error_analysis_M2M100_300_v1.csv")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.714151Z","iopub.execute_input":"2021-12-21T12:36:19.714429Z","iopub.status.idle":"2021-12-21T12:36:19.72223Z","shell.execute_reply.started":"2021-12-21T12:36:19.714405Z","shell.execute_reply":"2021-12-21T12:36:19.721226Z"}}
subset_values = [text.split(' | ')[-1] for text in subset_texts]
print(f"Length of subset keys: {len(subset_values)} and example: {subset_values[1]}")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.723861Z","iopub.execute_input":"2021-12-21T12:36:19.724542Z","iopub.status.idle":"2021-12-21T12:36:19.738359Z","shell.execute_reply.started":"2021-12-21T12:36:19.724459Z","shell.execute_reply":"2021-12-21T12:36:19.737299Z"}}
tr_subset_values = [text.split('|')[-1].strip() for text in tr_subset_texts]
print(f"Length of subset keys: {len(tr_subset_values)} and example: {tr_subset_values[1]}")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.739789Z","iopub.execute_input":"2021-12-21T12:36:19.740135Z","iopub.status.idle":"2021-12-21T12:36:19.753331Z","shell.execute_reply.started":"2021-12-21T12:36:19.7401Z","shell.execute_reply":"2021-12-21T12:36:19.75251Z"}}
bt_subset_values = []
for i, text in enumerate(bt_subset_texts):
    try:
        bt_value =  text.split('|')[-1].strip()
    except:
        print(i)
        bt_value = "None"
    bt_subset_values.append(bt_value)
print(f"Length of subset keys: {len(bt_subset_values)} and example: {bt_subset_values[1]}")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.756315Z","iopub.execute_input":"2021-12-21T12:36:19.756635Z","iopub.status.idle":"2021-12-21T12:36:19.792704Z","shell.execute_reply.started":"2021-12-21T12:36:19.756591Z","shell.execute_reply":"2021-12-21T12:36:19.791817Z"}}
# saving the context keys
bt_values_df = pd.DataFrame.from_dict({"original_values":subset_values,
                                      "translated_values": tr_subset_values,
                                      "backtranslated_values":bt_subset_values
                                     })

bt_values_df.to_csv("./ZH_context_values_backtranslation_error_analysis_M2M100_300_v1.csv")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.794233Z","iopub.execute_input":"2021-12-21T12:36:19.794634Z","iopub.status.idle":"2021-12-21T12:36:19.899414Z","shell.execute_reply.started":"2021-12-21T12:36:19.794594Z","shell.execute_reply":"2021-12-21T12:36:19.897767Z"}}
lev_dis = []

for i in range(len(subset_keys)):
  ratio = fuzz.ratio(str(subset_keys[i]), str(bt_subset_keys[i]))
  err_ratio = 1 - (ratio/100)
  dist = err_ratio * (len(str(subset_keys[i])) if len(str(subset_keys[i])) > len(str(bt_subset_keys[i])) else len(str(bt_subset_keys[i])))
  lev_dis.append(dist)
    
bt_keys_df['levenshtein_dist'] = lev_dis

temp = len(bt_keys_df[bt_keys_df["levenshtein_dist"]<3])
print(f"Small lev dist keys: {temp} ")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.900756Z","iopub.execute_input":"2021-12-21T12:36:19.901088Z","iopub.status.idle":"2021-12-21T12:36:19.945538Z","shell.execute_reply.started":"2021-12-21T12:36:19.90105Z","shell.execute_reply":"2021-12-21T12:36:19.944543Z"}}
lev_dis = []

for i in range(len(subset_values)):
  ratio = fuzz.ratio(str(subset_values[i]), str(bt_subset_values[i]))
  err_ratio = 1 - (ratio/100)
  dist = err_ratio * (len(str(subset_values[i])) if len(str(subset_values[i])) > len(str(bt_subset_values[i])) else len(str(bt_subset_values[i])))
  lev_dis.append(dist)
    
bt_values_df['levenshtein_dist'] = lev_dis

temp = len(bt_values_df[bt_values_df["levenshtein_dist"]<3])
print(f"Small lev dist keys: {temp} ")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:19.94692Z","iopub.execute_input":"2021-12-21T12:36:19.947285Z","iopub.status.idle":"2021-12-21T12:36:30.098358Z","shell.execute_reply.started":"2021-12-21T12:36:19.947234Z","shell.execute_reply":"2021-12-21T12:36:30.097488Z"}}
wordnet_similarity_score = []
for i in trange(len(subset_keys)):
  try: 
    syn_og = wordnet.synsets(subset_keys[i])[0]
  except:
    wordnet_similarity_score.append(0)
    continue

  try:
    syn_bt = wordnet.synsets(bt_subset_keys[i])[0]
  except:
    wordnet_similarity_score.append(0)
    continue
  
  ss = wordnet.wup_similarity(syn_og, syn_bt)
  wordnet_similarity_score.append(ss)

bt_keys_df['wordnet_similarity_score'] = wordnet_similarity_score

temp = len(bt_keys_df[ bt_keys_df["wordnet_similarity_score"] > 0 ])
print(f"High wordnet sim score: {temp}")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:30.099738Z","iopub.execute_input":"2021-12-21T12:36:30.100308Z","iopub.status.idle":"2021-12-21T12:36:38.347246Z","shell.execute_reply.started":"2021-12-21T12:36:30.100254Z","shell.execute_reply":"2021-12-21T12:36:38.346353Z"}}
wordnet_similarity_score = []
for i in trange(len(subset_values)):
  try: 
    syn_og = wordnet.synsets(subset_values[i])[0]
  except:
    wordnet_similarity_score.append(0)
    continue

  try:
    syn_bt = wordnet.synsets(bt_subset_values[i])[0]
  except:
    wordnet_similarity_score.append(0)
    continue
  
  ss = wordnet.wup_similarity(syn_og, syn_bt)
  wordnet_similarity_score.append(ss)

bt_values_df['wordnet_similarity_score'] = wordnet_similarity_score

temp = len(bt_values_df[ bt_values_df["wordnet_similarity_score"] > 0 ])
print(f"High wordnet sim score: {temp}")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:38.348591Z","iopub.execute_input":"2021-12-21T12:36:38.349103Z","iopub.status.idle":"2021-12-21T12:36:49.254989Z","shell.execute_reply.started":"2021-12-21T12:36:38.349063Z","shell.execute_reply":"2021-12-21T12:36:49.254041Z"}}
!pip install sentence-transformers

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T12:36:49.259399Z","iopub.execute_input":"2021-12-21T12:36:49.259705Z","iopub.status.idle":"2021-12-21T12:37:18.363482Z","shell.execute_reply.started":"2021-12-21T12:36:49.259675Z","shell.execute_reply":"2021-12-21T12:37:18.362621Z"}}
from sentence_transformers import SentenceTransformer, util
sentence_model = SentenceTransformer('paraphrase-mpnet-base-v2').to(device)

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2021-12-21T12:37:18.366492Z","iopub.execute_input":"2021-12-21T12:37:18.366787Z","iopub.status.idle":"2021-12-21T12:48:23.467162Z","shell.execute_reply.started":"2021-12-21T12:37:18.366761Z","shell.execute_reply":"2021-12-21T12:48:23.466282Z"}}
paraphrase_score = []

for i in trange(len(subset_keys)): 
  og_embed = sentence_model.encode(str(subset_keys[i]))
  bt_embed = sentence_model.encode(str(bt_subset_keys[i]))
  cos_sim = util.cos_sim(og_embed, bt_embed)
  paraphrase_score.append(cos_sim.item())

bt_keys_df['paraphrase_score'] = paraphrase_score

temp = len(bt_keys_df[bt_keys_df['paraphrase_score'] > 0.5])
print(f"high paraphrase score: {temp}")

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2021-12-21T12:48:23.468648Z","iopub.execute_input":"2021-12-21T12:48:23.469202Z","iopub.status.idle":"2021-12-21T12:59:31.330239Z","shell.execute_reply.started":"2021-12-21T12:48:23.469163Z","shell.execute_reply":"2021-12-21T12:59:31.329186Z"}}
paraphrase_score = []

for i in trange(len(subset_values)): 
  og_embed = sentence_model.encode(str(subset_values[i]))
  bt_embed = sentence_model.encode(str(bt_subset_values[i]))
  cos_sim = util.cos_sim(og_embed, bt_embed)
  paraphrase_score.append(cos_sim.item())

bt_values_df['paraphrase_score'] = paraphrase_score

temp = len(bt_values_df[bt_values_df['paraphrase_score'] > 0.5])
print(f"high paraphrase score: {temp}")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T13:05:44.256744Z","iopub.execute_input":"2021-12-21T13:05:44.257093Z","iopub.status.idle":"2021-12-21T13:05:44.342647Z","shell.execute_reply.started":"2021-12-21T13:05:44.257053Z","shell.execute_reply":"2021-12-21T13:05:44.341772Z"}}
bt_keys_df.to_csv("./ZH_context_keys_backtranslation_error_analysis_M2M100_300_v2.csv")
bt_values_df.to_csv("./ZH_context_values_backtranslation_error_analysis_M2M100_300_v2.csv")

# %% [code]
