#!/usr/bin/env python
# coding: utf-8

# In[1]:
import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

import wandb
import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
import os
cache_dir = os.makedirs("cache",exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = "cache"
os.environ['HF_DATASETS_CACHE'] = "cache"

# In[2]:


import pandas as pd
import os


# In[3]:


from simpletransformers.seq2seq import Seq2SeqModel,Seq2SeqArgs


# In[4]:


Seq2SeqArgs()


# In[5]:


model_args = Seq2SeqArgs()


# In[6]:

wandb.init(project="cs-en-th-transformer")
model_args.num_train_epochs = 5
model_args.logging_steps=1
model_args.wandb_kwargs= {"job_type": "training"}
model_args.use_multiprocessing=False
model_args.dataloader_num_workers=0
model_args.use_multiprocessing_for_evaluation=False
model_args.dataset_cache_dir="dataset_cache/"
model_args.overwrite_output_dir = True
#model_args.evaluate_generated_text = True
#model_args.evaluate_during_training = True
#model_args.evaluate_during_training_verbose = True


# In[7]:


model_args.wandb_project="cs-en-th-transformer"


# In[ ]:





# In[8]:


# epochs=50
# learning_rate=0.001


# In[9]:


# model_args ={
#     "reprocess_input_data": True,
#     "overwrite_output_dir": True,
#     "num_train_epochs":epochs,
#     'learning_rate':learning_rate,
#     'wandb_project': "simpletransformers"
# }


# In[10]:


# model = Seq2SeqModel(
#     "bert",
#     "bert-base-multilingual-cased",#"monsoon-nlp/bert-base-thai",
#     "bert-base-multilingual-cased",#"monsoon-nlp/bert-base-thai",
#     args=model_args
# )
model = Seq2SeqModel(
    encoder_decoder_type="marian",
    encoder_decoder_name="ep4/",
    args=model_args,
    use_cuda=True,
)


# In[11]:


path_dataset = os.path.join('..','dataset','cs')

train_filepaths = os.path.join(path_dataset,'train.tsv')
dev_filepaths = os.path.join(path_dataset,'dev.tsv')
test_filepaths = os.path.join(path_dataset,'test.tsv')

train_df = pd.read_csv(train_filepaths,sep="\t")
dev_df = pd.read_csv(dev_filepaths,sep="\t")
test_df = pd.read_csv(test_filepaths,sep="\t")


# In[12]:


def load_data(data_path):
    new_df=pd.DataFrame()
    new_df["target_text"]=[str(i) for i in data_path['word']]
    new_df["input_text"] =[str(i) for i in data_path['roman']]
    return new_df


# In[13]:


train_df=load_data(train_df)


# In[14]:


train_df


# In[15]:


dev_df=load_data(dev_df)


# In[16]:


dev_df


# In[ ]:


model.train_model(train_df)#,eval_data=dev_df)


# In[ ]:





# In[ ]:




