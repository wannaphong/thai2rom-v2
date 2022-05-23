#!/usr/bin/env python
# coding: utf-8

# # Thai to English Transliteration with Seq2Seq model

# In[1]:


#get_ipython().system('nvidia-smi')


# In[2]:


import os
os.environ['CUDA_VISIBLE_DEVICES']='6'


# In[3]:


import time
import sys
import os
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
from matplotlib import pyplot as plt
from collections import OrderedDict

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[4]:


# !pip install wandb --upgrade


# In[11]:


import pandas as pd


# In[5]:


import wandb

# start a new experiment
wandb.init(project="thai_romanize_pytorch_seq2seq_attention")


# In[6]:


# # Check if GPUs are in the machine, otherwise assign device as CPU
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device


# In[7]:


# !git clone https://github.com/artificiala/thai-romanization.git


# In[8]:




# The csv file contains two columns indicates Thai text and its corresponding English tranliteration

# In[9]:


# impdevice


# In[12]:


path_dataset = os.path.join('..','dataset','thai2rom')

train_filepaths = os.path.join(path_dataset,'train.tsv')
dev_filepaths = os.path.join(path_dataset,'dev.tsv')
test_filepaths = os.path.join(path_dataset,'test.tsv')

train_df = pd.read_csv(train_filepaths,sep="\t")
dev_df = pd.read_csv(dev_filepaths,sep="\t")
test_df = pd.read_csv(test_filepaths,sep="\t")


# In[13]:


def load_data(data_path):
    input_texts = list(data_path['word'])
    target_texts = list(data_path['roman'])
    return input_texts, target_texts


# In[14]:


input_texts, target_texts = load_data(train_df)


# In[15]:


input_texts[0]


# In[16]:


# Define special characters
UNK_token = '<UNK>'
PAD_token = '<PAD>'
START_token = '<start>'
END_token = '<end>'
MAX_LENGTH = 60

class Language:
    def __init__(self, name, is_input=False):
        self.name = name
        self.characters = set()
        self.n_chars = 0
        self.char2index = {}
        self.index2char = {}

        if is_input == True:
            self.index2char = { 0: PAD_token, 1: UNK_token, 2: START_token, 3: END_token }
            self.char2index = { ch:i for i, ch in self.index2char.items() } #reverse dictionary
            self.n_chars = 4
        else:
            self.index2char = { 0: PAD_token, 1: START_token, 2: END_token }
            self.char2index = { ch:i for i, ch in self.index2char.items() } #reverse dictionary
            self.n_chars = 3

    def addText(self, text):
        for character in text:
            self.addCharacter(character)
    
    def addCharacter(self, character):
        if character not in self.char2index.keys():
            self.char2index[character] = self.n_chars
            self.index2char[self.n_chars] = character
            self.n_chars += 1
            
            
def indexesFromText(lang, text):
    """returns indexes for all character given the text in the specified language"""
    return [lang.char2index[char] for char in text]

def tensorFromText(lang, text):
    """construct a tensor given the text in the specified language"""
    indexes = indexesFromText(lang, text)
    indexes.append(lang.char2index[END_token])
    
    no_padded_seq_length = len(indexes) # Number of characters in the text (including <END> token)
    # Add padding token to make all tensors in the same length
    for i in range(len(indexes), MAX_LENGTH): # padding
        indexes.append(lang.char2index[PAD_token])
        
    return torch.tensor(indexes, dtype=torch.long), no_padded_seq_length

def filterPair(p1, p2):
    """filter for the pair the both texts has length less than `MAX_LENGTH`"""
    return len(p1) < MAX_LENGTH and len(p2) < MAX_LENGTH

def tensorsFromPair(pair, lang1, lang2):
    """construct two tensors from a pair of source and target text specified by source and target language"""
    input_tensor, input_length = tensorFromText(lang1, pair[0])
    target_tensor, target_length = tensorFromText(lang2, pair[1])
    return input_tensor, target_tensor, input_length, target_length



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_text, target_text, lang_th, lang_th_romanized = sample['input_text'], sample['target_text'],                                                               sample['lang_th'], sample['lang_th_romanized']

        input_tensor, target_tensor, input_length, target_length = tensorsFromPair([input_text, target_text], 
                                                                                   lang_th, 
                                                                                   lang_th_romanized)
        
        return {
                'input_text': input_text,
                'target_text': target_text,
                'input_length': input_length,
                'target_length': target_length,
                'input_tensor': input_tensor,
                'target_tensor': target_tensor
               }
    
    
class ThaiRomanizationDataset(Dataset):
    """Thai Romanization Dataset class"""
    def __init__(self, 
                 data_path, 
                 transform=transforms.Compose([ ToTensor() ])):

        input_texts, target_texts = load_data(data_path)
        
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.transform = transform
        self.lang_th = None
        self.lang_th_romanized = None
        self.counter = Counter()
        self.pairs = []
        self.prepareData()

    def prepareData(self):
        self.lang_th = Language('th', is_input=True)
        self.lang_th_romanized = Language('th_romanized', is_input=False)
        for i in range(len(self.input_texts)):
            
            input_text = str(self.input_texts[i])
            target_text = str(self.target_texts[i])
            
            # Count the number of input and target sequences with length `x`
            self.counter.update({ 
                                  'len_input_{}'.format(len(input_text)): 1, 
                                  'len_target_{}'.format(len(target_text)): 1 
                                })
            
            if filterPair(input_text, target_text):
                self.pairs.append((input_text, target_text))
                self.lang_th.addText(input_text)
                self.lang_th_romanized.addText(target_text)    

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        
        sample = dict()
        sample['input_text'] = self.pairs[idx][0]
        sample['target_text'] = self.pairs[idx][1]
        
        sample['lang_th'] = self.lang_th
        sample['lang_th_romanized'] = self.lang_th_romanized

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[17]:


thai_romanization_dataset = ThaiRomanizationDataset(train_df)


# In[ ]:





# In[18]:


thai_romanization_dataset.lang_th_romanized.index2char


# ## Seq2Seq Model architecture

# ## 1. Encoder

# Encoder 
#     - Embedding layer :(vocaburay_size, embedding_size) 
#         Input: (batch_size, sequence_length)
#         Output: (batch_size, sequence_length, embebeding_size)
#       
#     - Bi-LSTM layer : (input_size, hidden_size, num_layers, batch_first=True)
#         Input: (input=(batch_size, seq_len, embebeding_size),  hidden)
#         Output: (output=(batch_size, seq_len, hidden_size),
#                  (h_n, c_n))
#      
#      
# __Steps:__
# 
# 1. Receives a batch of source sequences (batch_size, MAX_LENGTH) and a 1-D array of the length for each sequence (batch_size).
#      
# 2. Sort sequences in the batch by sequence length (number of tokens in the sequence where <PAD> token is excluded).
# 
# 3. Feed the batch of sorted sequences into the Embedding Layer to maps source character indices into vectors. (batch_size,  sequence_length, embebeding_size)
# 
# 4. Use `pack_padded_sequence` to let LSTM packed input with same length at time step $t$ together. This will reduce time required for training by avoid feeding `<PAD>` token to the LSTMs.
# 
# 
# 5. Returns LSTM outputs in the unsorted order, and the LSTM hidden state vectors.
#      

# In[19]:




class Encoder(nn.Module):
    
    def __init__(self, vocabulary_size, embedding_size, hidden_size, dropout=0.5):
        """Constructor"""
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.character_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, 
                            hidden_size=hidden_size // 2, 
                            bidirectional=True,
                            batch_first=True)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, sequences, sequences_lengths):
        batch_size = sequences.size(0)
        self.hidden = self.init_hidden(batch_size) # batch_size

        # sequences :(batch_size, sequence_length=MAX_LENGTH)
        # sequences_lengths: (batch_size)  # an 1-D indicating length of each sequence (excluded <PAD> token) in `seq`
        
        # 1. Firstly we sort `sequences_lengths` according to theirs values and keep list of indexes to perform sorting
        sequences_lengths = np.sort(sequences_lengths)[::-1] # sort in ascending order and reverse it
        index_sorted = np.argsort(-sequences_lengths) # use negation in sort in descending order
        index_unsort = np.argsort(index_sorted) # to unsorted sequence
        
        
        # 2. Then, we change position of sequence in `sequences` according to `index_sorted`
        index_sorted = torch.from_numpy(index_sorted)
        sequences = sequences.index_select(0, index_sorted)
        
        # 3. Feed to Embedding Layer
        
        sequences = self.character_embedding(sequences)
        sequences = self.dropout(sequences)
        
#         print('sequences',sequences.size(), sequences)
            
        # 3. Use function: pack_padded_sequence to let LSTM packed input with same length at time step T together
        
        # Quick fix: Use seq_len.copy(), instead of seq_len to fix `Torch.from_numpy not support negative strides`
        # ndarray.copy() will alocate new memory for numpy array which make it normal, I mean the stride is not negative any more.

        sequences_packed = nn.utils.rnn.pack_padded_sequence(sequences,
                                                             sequences_lengths.copy(),
                                                             batch_first=True)
#         print('sequences_packed', sequences_packed)

        # 4. Feed to LSTM
        sequences_output, self.hidden = self.lstm(sequences_packed, self.hidden)
        
        # 5. Unpack
        sequences_output, _ = nn.utils.rnn.pad_packed_sequence(sequences_output, batch_first=True)

        # 6. Un-sort by length
        index_unsort = torch.from_numpy(index_unsort)
        sequences_output = sequences_output.index_select(0, Variable(index_unsort))

#         print('hidden shape', self.hidden[0].shape, self.hidden[0], self.hidden[1].shape, self.hidden[1])
        return sequences_output, self.hidden
    
    def init_hidden(self, batch_size):
        h_0 = torch.zeros([2, batch_size, self.hidden_size // 2], requires_grad=True)
        c_0 = torch.zeros([2, batch_size, self.hidden_size // 2], requires_grad=True)
        
        return (h_0, c_0)
    
def save_model(name, epoch, loss, model):
    print('Save model at epoch ', epoch)
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'char_to_ix': thai_romanization_dataset.lang_th.char2index,
        'ix_to_char': thai_romanization_dataset.lang_th.index2char,
        'target_char_to_ix': thai_romanization_dataset.lang_th_romanized.char2index,
        'ix_to_target_char':thai_romanization_dataset.lang_th_romanized.index2char,
        'encoder_params': (INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, ENC_DROPOUT), 
        'decoder_params': (OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, DEC_DROPOUT)
        
    }, "{}.best_epoch-{}.tar".format(name, epoch))
    
 
    
def load_model(model_path):
    
    data = torch.load(model_path)
    
    INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, ENC_DROPOUT = data['encoder_params']
    OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, DEC_DROPOUT = data['decoder_params']

    
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM)
    decoder = AttentionDecoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM)

    model = Seq2Seq(encoder, decoder)
    
    model.load_state_dict(data['model_state_dict'])
    
    
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(data['optimizer_state_dict'])
    criterion = nn.CrossEntropyLoss(ignore_index = 0)

    char_to_ix = data['char_to_ix']
    ix_to_char = data['ix_to_char'] 
    target_char_to_ix = data['target_char_to_ix']
    ix_to_target_char = data['ix_to_target_char']
    
    
    return {
        'model': model,
        'optmizer': optimizer,
        'char_to_ix': char_to_ix,
        'ix_to_char' : ix_to_char,
        'target_char_to_ix': target_char_to_ix,
        'ix_to_target_char': ix_to_target_char
    }
       


# ## Decoder

#    
# Decoder architecture
# 
#     - Embedding layer :(vocabulary_size, embebeding_size)
#         Input: (batch_size, sequence_length=1)
#         Output: (batch_size, sequence_length=1, embebeding_size)
#     - RNN layer :input_size=embebeding_size, hidden_size, num_layers, batch_first=True)
#         Input: (input=(batch_size, input_size=embedding_dimension), hidden:tuple=encoder_hidden
#         Output: (batch_size, seq_len, hidden_size), (h_n, c_n)
#     - Attention Layer: (in_features=hidden_size, out_features=hidden_size, bias=True)
#     - Linear Layer: (in_features, out_features=vocabulary_size)
#         Input: (batch_size, hidden_size)
#         Output: (batch_size, vocabulary_size)
#     
#     - Softmax layer
#         Input: (batch_size, vocabulary_size)
#         Output: (batch_size, vocabulary_size)
# 
# 
# 
# For the Attention mechanishm in the Decoder, Luong-style attention [[Luong et. al (2015)](https://arxiv.org/abs/1508.04025)] is used. 
# 
# 
# 
# __Steps:__
# 
# 1. Receives a batch of <start> token (batch_size, 1) and a batch of Encoder's hidden state.
#      
# 2. Embed input into vectors.
# 
# 3. Feed vectors from (2) to the LSTM.
# 
# 4. Feed the output of LSTM at time step $t_1$ and Encoder output to the Attention Layer.
# 
# 5. Attention layer, returns weights for Encoder's hidden states in every time step (masked out the time step with <PAD> token), then multiply with Encoder's hidden states to obtain a context vector
#     
# 6. Concatenate both decoder hidden state and the context vector, feed to a linear layer, and return its output.
# 
# 7. Decoder then returns, final output, decoder's hidden state, attention weights, and context vector at time step $t$

# In[20]:



class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: B x 1 x h ; 
        # encoder_outputs: B x S x h

        # Calculate energies for each encoder output
        if self.method == 'dot':
            attn_energies = torch.bmm(encoder_outputs, hidden.transpose(1, 2)).squeeze(2)  # B x S
        elif self.method == 'general':
            attn_energies = self.attn(encoder_outputs.view(-1, encoder_outputs.size(-1)))  # (B * S) x h
            attn_energies = torch.bmm(attn_energies.view(*encoder_outputs.size()),
                                      hidden.transpose(1, 2)).squeeze(2)  # B x S
        elif self.method == 'concat':
            attn_energies = self.attn(
                torch.cat((hidden.expand(*encoder_outputs.size()), encoder_outputs), 2))  # B x S x h
            attn_energies = torch.bmm(attn_energies,
                                      self.other.unsqueeze(0).expand(*hidden.size()).transpose(1, 2)).squeeze(2)

        attn_energies = attn_energies.masked_fill(mask == 0, -1e10)

        # Normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies, 1)

class AttentionDecoder(nn.Module): 
    
    def __init__(self, vocabulary_size, embedding_size, hidden_size, dropout=0.5):
        """Constructor"""
        super(AttentionDecoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.character_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size + self.hidden_size,
                            hidden_size=hidden_size,
                            bidirectional=False,
                            batch_first=True)
        
        self.attn = Attn(method="general", hidden_size=self.hidden_size)
        self.linear = nn.Linear(hidden_size * 2, vocabulary_size)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, input, last_hidden, last_context, encoder_outputs, mask):
        """"Defines the forward computation of the decoder"""
        # input: (B, 1) ,
        # last_hidden: (num_layers * num_directions, B, hidden_dim)
        # last_context: (B, 1, hidden_dim)
        # encoder_outputs: (B, S, hidden_dim)
        
        embedded = self.character_embedding(input)
        embedded = self.dropout(embedded)
        
        # embedded: (batch_size, emb_dim)
        rnn_input = torch.cat((embedded, last_context), 2)

        output, hidden = self.lstm(rnn_input, last_hidden)        
        attn_weights = self.attn(output, encoder_outputs, mask)  # B x S
    
        #  context = (B, 1, S) x (B, S, hidden_dim)
        #  context = (B, 1, hidden_dim)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  
        
        output = torch.cat((context.squeeze(1), output.squeeze(1)), 1)
        output = self.linear(output)
        
        return output, hidden, context, attn_weights


# ## Seq2Seq model
# 
# This class encapsulate _Decoder_ and _Encoder_ class.
# 
# __Steps:__
# 
# 1. The input sequcence $X$ is fed into the encoder to receive one hidden state vector.
# 
# 2. The initial decoder hidden state is set to be the hidden state vector of the encoder
# 
# 3. Add a batch of `<start>` tokens (batch_size, 1) as the first input $y_1$
#     
# 4. Then, decode within a loop:
#     - Inserting the input token $y_t$, previous hidden state, $s_{t-1}$, and the context vector $z$ into the decoder
#     - Receiveing a prediction $\hat{y}$ and a new hidden state $s_t$
#     - Then, either use teacher forcing to let groundtruth target character as the input for the decoder at time step $t+1$, or let the result from decoder as the input for the next time step.

# In[21]:


class Seq2Seq(nn.Module): 

    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = 0

        assert encoder.hidden_size == decoder.hidden_size
    
    def create_mask(self, source_seq):
        mask = (source_seq != self.pad_idx)
        return mask
        
  
    def forward(self, source_seq, source_seq_len, target_seq, teacher_forcing_ratio = 0.5):
        """
            Parameters:
                - source_seq: (batch_size x MAX_LENGTH) 
                - source_seq_len: (batch_size x 1)
                - target_seq: (batch_size x MAX_LENGTH)

            Returns
        """
        batch_size = source_seq.size(0)
        start_token = thai_romanization_dataset.lang_th_romanized.char2index["<start>"]
        end_token = thai_romanization_dataset.lang_th_romanized.char2index["<end>"]
        max_len = MAX_LENGTH
        target_vocab_size = self.decoder.vocabulary_size

        # init a tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, target_vocab_size)
        
        if target_seq is None:
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            inference = True
        else:
            inference = False

    
        # feed mini-batch source sequences into the `Encoder`
        encoder_outputs, encoder_hidden = self.encoder(source_seq, source_seq_len)

        # create a Tensor of first input for the decoder
        decoder_input = torch.tensor([[start_token] * batch_size]).view(batch_size, 1)
        
        # Initiate decoder output as the last state encoder's hidden state
        decoder_hidden_0 = torch.cat([encoder_hidden[0][0], encoder_hidden[0][1]], dim=1).unsqueeze(dim=0)
        decoder_hidden_1 = torch.cat([encoder_hidden[1][0], encoder_hidden[1][1]], dim=1).unsqueeze(dim=0)
        decoder_hidden = (decoder_hidden_0, decoder_hidden_1) # (hidden state, cell state)

        # define a context vector
        decoder_context = Variable(torch.zeros(encoder_outputs.size(0), encoder_outputs.size(2))).unsqueeze(1)
        
        max_source_len = encoder_outputs.size(1)
        mask = self.create_mask(source_seq[:, 0:max_source_len])
            
       
        for di in range(max_len):
            decoder_output, decoder_hidden, decoder_context, attn_weights = self.decoder(decoder_input,
                                                                                    decoder_hidden,
                                                                                    decoder_context,
                                                                                    encoder_outputs,
                                                                                    mask)
            # decoder_output: (batch_size, target_vocab_size)

            topv, topi = decoder_output.topk(1)
            outputs[di] = decoder_output
    
            teacher_force = random.random() < teacher_forcing_ratio


            decoder_input = target_seq[:, di].reshape(batch_size, 1) if teacher_force else topi.detach() 

            if inference and decoder_input == end_token:
                return outputs[:di]
        return outputs


# Initializae model
# 

# In[23]:


SEED = 0
BATCH_SIZE = 256

TRAIN_RATIO = 0.8

N = len(thai_romanization_dataset)

print('Number of samples: ', N)

# np.random.seed(SEED)
# np.random.shuffle(indices)
# train_indices, val_indices = indices[:train_split_idx], indices[train_split_idx:]

# print('train_indices', train_indices[0:5])
# print('val_indices', val_indices[0:5])


# In[24]:


from tqdm import tqdm


# In[25]:


val_indices = []
for i in tqdm(list(dev_df['word'])):
    val_indices.append(input_texts.index(i))


# In[26]:


val_indices[0]


# In[27]:


input_texts[val_indices[0]]


# In[28]:


input_texts[val_indices[-1]]


# In[31]:


train_indices = [i for i in tqdm(list(range(N)))]


# In[32]:


train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
                                   
train_dataset_loader = torch.utils.data.DataLoader(
                                             thai_romanization_dataset,
                                             batch_size=BATCH_SIZE, 
                                             sampler=train_sampler,
                                             num_workers=0)

val_dataset_loader = torch.utils.data.DataLoader(
                                             thai_romanization_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             sampler=valid_sampler,
                                             num_workers=0)


print('Number of train mini-batches', len(train_dataset_loader))
print('Number of val mini-batches', len(val_dataset_loader))


# In[33]:


INPUT_DIM = len(thai_romanization_dataset.lang_th.char2index)
OUTPUT_DIM = len(thai_romanization_dataset.lang_th_romanized.char2index)

ENC_EMB_DIM = 128
ENC_HID_DIM = 256
ENC_DROPOUT = 0.5

DEC_EMB_DIM = 128
DEC_HID_DIM = 256
DEC_DROPOUT = 0.5

_encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM)
_decoder = AttentionDecoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM)

model = Seq2Seq(_encoder, _decoder)


# In[ ]:


# data = load_model('./thai2rom-pytorch.attn.best_epoch-10.tar')

# _model


# In[34]:


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)


# In[35]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[36]:



learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index = 0)


# In[ ]:





# ## Training

# In[37]:


N_EPOCHS = 200


# In[38]:


wandb.config = {"learning_rate": learning_rate, "epochs": N_EPOCHS, "batch_size": BATCH_SIZE}


# In[39]:


wandb.watch(model, log_freq=100)


# In[40]:


print_loss_every = 100
# teacher_forcing_ratio = 0.0

def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio=0.5):
    
    model.train()
    
    epoch_loss = 0
    for i, batch in tqdm(enumerate(iterator), total = len(iterator)):
        optimizer.zero_grad()

        source_seq, source_seq_len = batch['input_tensor'], batch['input_length']
        batch_size = source_seq.size(0)
        
        # target_seq: (batch_size , MAX_LENGTH)
        # output: (MAX_LENGTH , batch_size , target_vocab_size)
        target_seq = batch['target_tensor']

        output = model(source_seq, source_seq_len, target_seq, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # target_seq -> (MAX_LENGTH , batch_size)
        target_seq = target_seq.transpose(0, 1)

        # target_seq -> ((MAX_LENGTH - 1) * batch_size)
        target_seq = target_seq[1:].contiguous().view(-1)

        # output -> ((MAX_LENGTH -1) * batch_size, target_vocab_size)        
        output = output[1:].view(-1, output.shape[-1])

        loss = criterion(output, target_seq)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        wandb.log({"train_loss":loss.item()})
        
        epoch_loss += loss.item()
        
        
    return epoch_loss / len(iterator)


# In[41]:


char_2_ix, ix_2_char, target_char_to_ix, ix_to_target_char = thai_romanization_dataset.lang_th.char2index ,                                                              thai_romanization_dataset.lang_th.index2char ,                                                              thai_romanization_dataset.lang_th_romanized.char2index ,                                                             thai_romanization_dataset.lang_th_romanized.index2char


# In[42]:


def evaluate(model, iterator, criterion):
    
    model.eval()

    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            source_seq, source_seq_len = batch['input_tensor'], batch['input_length']
            batch_size = source_seq.size(0)

            # target_seq: (batch_size , MAX_LENGTH)
            # output: (MAX_LENGTH , batch_size , target_vocab_size)
            target_seq = batch['target_tensor']
            output = model(source_seq, source_seq_len, target_seq)
        
            # target_seq -> (MAX_LENGTH , batch_size)
            target_seq = target_seq.transpose(0, 1)

            # target_seq -> ((MAX_LENGTH - 1) * batch_size)
            target_seq = target_seq[1:].contiguous().view(-1)

            # output -> ((MAX_LENGTH -1) * batch_size, target_vocab_size)        
            output = output[1:].view(-1, output.shape[-1])

            loss = criterion(output, target_seq)
            wandb.log({"evaluate_loss":loss.item()})
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def inference(model, text, char_2_ix, ix_2_char, target_char_to_ix, ix_to_target_char):
    model.eval()

    input_seq =  [ch for ch in text] +  ['<end>']
    numericalized = [char_2_ix[ch] for ch in input_seq] 
    
#     print('input ',numericalized)
    sentence_length = [len(numericalized)]

    tensor = torch.LongTensor(numericalized).view(1, -1)
    
#     print(tensor)
    translation_tensor_logits = model(tensor, sentence_length, None, 0) 
#     print(translation_tensor_logits)
    try:
        translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1).cpu().numpy()
        translation_indices = [t for t in translation_tensor]
        
#         print('translation_tensor', translation_tensor)
        translation = [ix_to_target_char[t] for t in translation_tensor]
    except:
        translation_indices = [0]
        translation = ['<pad>']
    return ''.join(translation), translation_indices

def show_inference_example(model, input_texts, target_texts, char_2_ix, ix_2_char, target_char_to_ix, ix_to_target_char):
    for index, input_text in enumerate(input_texts):
        prediction, indices = inference(model, input_text, char_2_ix, ix_2_char, target_char_to_ix, ix_to_target_char)
        print('groundtruth: {}'.format(target_texts[index]))
        print(' prediction: {} {}\n'.format(prediction, indices))
        

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    


# In[43]:


print(char_2_ix)


# In[44]:


print(ix_to_target_char)


# In[45]:


# _model.state_dict()


# In[46]:


show_inference_example(model, ['การ'], [''], char_2_ix, ix_2_char, target_char_to_ix, ix_to_target_char)


# In[47]:


from distance import levenshtein


# In[48]:


def calc_per(Y_true, Y_pred):
    '''Calc phoneme error rate
    Y_true: list of predicted phoneme sequences. e.g., [["k", "a", "m", "a", "n", "d"], ...]
    Y_pred: list of ground truth phoneme sequences. e.g., [["k", "a", "m", "a", "n", "d"], ...]
    '''
    num_phonemes, num_erros = 0, 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        num_phonemes += len(y_true)
        num_erros += levenshtein(y_true, y_pred)

    per = round(num_erros / num_phonemes, 4)
    return per


# In[49]:


calc_per([["k", "a", "m", "a", "n", "d"]], [["k", "a", "m", "a", "n", "d"]])


# In[50]:


# Functions for model performance evaluation
def precision(pred_chars, target_chars):
    # TP / TP + FP
    pred_chars_multiset = Counter(pred_chars)
    target_chars_multiset = Counter(target_chars)

    overlap = list((pred_chars_multiset & target_chars_multiset).elements())
    n_overlap = len(overlap)

    return n_overlap / max(len(pred_chars), 1)

def recall(pred_chars, target_chars):
    # TP / TP + FN
        
    pred_chars_multiset = Counter(pred_chars)
    target_chars_multiset = Counter(target_chars)

    overlap = list((pred_chars_multiset & target_chars_multiset).elements())
    n_overlap = len(overlap)
    return n_overlap / len(target_chars)

def f1(precision, recall):
    
    return (2  * precision * recall) / (precision + recall)

def em(pred, target):
    if pred == target:
        return 1
    return 0

def em_char(pred_chars, target_chars):
    N_target_chars = len(target_chars)
    N_pred_chars = len(pred_chars)

    score = 0
    for index in range(min(N_pred_chars, N_target_chars)):
        if target_chars[index] == pred_chars[index]:
            score+=1
            
    return score / max(N_target_chars, N_pred_chars)


# In[51]:


def evaluate_inference(model, val_indices):
    cumulative_precision = 0
    cumulative_recall = 0
    cumulative_em = 0
    cumulative_em_char = 0
    
    N = len(val_indices)

    epoch_loss = 0
    prediction_results = []
    em_char_score = 0
    for i, val_index in tqdm(enumerate(val_indices), total=N):
        input_text = thai_romanization_dataset.input_texts[val_index]
        target_text = thai_romanization_dataset.target_texts[val_index]

        
        prediction, indices = inference(model, input_text, char_2_ix, ix_2_char, target_char_to_ix, ix_to_target_char)
        prediction_results.append(prediction)

        pred_chars = [char for char in prediction]
        target_chars = [char for char in target_text]

        cumulative_precision += precision(pred_chars, target_chars)
        cumulative_recall +=  recall(pred_chars, target_chars)
        cumulative_em_char += em_char(pred_chars, target_chars)
        cumulative_em += em(prediction, target_text)

    macro_average_precision = cumulative_precision / N
    macro_average_recall = cumulative_recall /N
    f1_macro_average = f1(macro_average_precision, macro_average_recall) 
    em_score = cumulative_em / N
    em_char_score = cumulative_em_char / N
    return f1_macro_average, em_score, em_char_score, prediction_results


# In[52]:


CLIP = 5

best_valid_loss = float('inf')

# for epoch in range(1, N_EPOCHS):
#     model.train()
#     start_time = time.time()
    
#     train_loss = train(model, train_dataset_loader, optimizer, criterion, CLIP, teacher_forcing_ratio=0.5)
#     valid_loss = evaluate(model, val_dataset_loader, criterion)
#     model.eval()
#     with torch.no_grad():
#         f1_macro_average, em_score, em_char_score, prediction_results=evaluate_inference(model, val_indices)
    

#     end_time = time.time()
#     save_model('thai2rom-pytorch-%s.attn.v6' % str(epoch), epoch, valid_loss, model)

#     wandb.log(
#         {
#             "epoch": epoch,
#             "epoch_train_loss": train_loss,
#             "epoch_valid_loss":valid_loss,
#             "valid_f1_macro_average":f1_macro_average,
#             "valid_em_score":em_score,
#             "valid_em_char_score":em_char_score,
#             "epoch_time":end_time-start_time
#         }
#     )
#     with open("pred/pred-%s"% str(epoch),"w",encoding="utf-8") as f:
#         f.write(str(prediction_results))
data=load_model("thai2rom-pytorch-72.attn.v6.best_epoch-72.tar")
model = data['model']
optimizer=data['optmizer']
for epoch in range(72+1, N_EPOCHS):
 model.train()
 start_time = time.time()
 train_loss = train(model, train_dataset_loader, optimizer, criterion, CLIP, teacher_forcing_ratio=0.5)
 valid_loss = evaluate(model, val_dataset_loader, criterion)
 end_time = time.time()
 save_model('new-model/thai2rom-pytorch-%s.attn.v6' % str(epoch), epoch, valid_loss, model)
 wandb.log({"epoch": epoch,"epoch_train_loss": train_loss,"epoch_valid_loss":valid_loss,"epoch_time":end_time-start_time})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


print(thai_romanization_dataset.input_texts[5000:5010])
print(thai_romanization_dataset.target_texts[5000:5010])


# In[ ]:





# 
# ## Evaluation on val_set with following metrics:
#    
# 1. F1-score (macro-average) -- Character level
# 
# 2. Exact Match (EM)
# 
# 3. Exact Match (EM) - Character level
# 
#     

# In[ ]:


# data = load_model('./thai2rom-pytorch.attn.v4.best_epoch-29.tar')

# _model = data['model']
# _model           

# char_to_ix = data['char_to_ix']
# ix_to_char =  data['ix_to_char']
# target_char_to_ix = data['target_char_to_ix']
# ix_to_target_char = data['ix_to_target_char']


# In[ ]:





# In[ ]:




