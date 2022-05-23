from collections import defaultdict
from operator import length_hint
import os
import re
from numpy import append
import pandas as pd
from pythainlp import thai_vowels, thai_consonants
from sklearn.model_selection import train_test_split

random_state_test = 42
test_size_test=0.2

list_thai_consonants = list(thai_consonants)
reg = "[ฯ%s]*" % thai_vowels

list_train = defaultdict(list)
list_dev = defaultdict(list)
list_test = defaultdict(list)

dict_by_c = defaultdict(list)

path = os.path.join(".","raw","data.tsv")

df = pd.read_csv(path,sep="\t")
#The *mask* variable is a dataframe of booleans, giving you True or False for the selected condition
mask = df[['word','roman']].applymap(lambda x: len(str(x)) <= 60)

#Here you can just use the mask to filter your rows, using the method *.all()* to filter only rows that are all True, but you could also use the *.any()* method for other needs
df = df[mask.all(axis=1)].dropna(how='all')

list_w_r = list(zip(df['word'],df['roman']))

for w,r in list_w_r:
    _temp = re.sub(reg,"",w)
    if _temp != "" and _temp!=None:
        dict_by_c[_temp[0]].append((w.replace(u'\xa0',' '),str(r).replace('-','').replace(u'\xa0',' ')))

for i in list_thai_consonants:
    _d = dict_by_c[i]
    length = len(_d)
    _train,_test=train_test_split(_d,random_state=random_state_test,test_size=test_size_test,shuffle=True)
    _train,_dev = train_test_split(_train,random_state=random_state_test,test_size=int(length*0.2),shuffle=True)
    for w,r in _train+_dev:
        list_train['word'].append(w)
        list_train['roman'].append(r)
    for w,r in _dev:
        list_dev['word'].append(w)
        list_dev['roman'].append(r)
    for w,r in _test:
        list_test['word'].append(w)
        list_test['roman'].append(r)

train_df = pd.DataFrame.from_dict(dict(list_train))
dev_df = pd.DataFrame.from_dict(dict(list_dev))
test_df = pd.DataFrame.from_dict(dict(list_test))

train_df.to_csv('train.tsv', sep='\t', index=False)
dev_df.to_csv('dev.tsv', sep='\t', index=False)
test_df.to_csv('test.tsv', sep='\t', index=False)