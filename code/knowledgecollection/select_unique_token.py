'''This version include inter cosine similarity'''


import torch
from torchtext.legacy import data
import random
import csv
from numpy import *
from transformers import BertTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer('vocab_update.txt', do_lower_case= True)
init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id


max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
print('max input length',max_input_length)
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

def data_loader(path,vocab_file,SEED):
    TEXT = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)
    LABEL = data.LabelField(dtype=torch.long)
    fields = [('text', TEXT), ('label', LABEL)]
    all_data, vocab_data = data.TabularDataset.splits(path=path,
                                                      train=vocab_file,

                                                      test=vocab_file,
                                                      format='csv',
                                                      fields=fields,
                                                      skip_header=True)
    vocab_train_data1, vocab_valid_data = vocab_data.split(random_state=random.seed(SEED), split_ratio=0.8)

    LABEL.build_vocab(all_data)

    texicon = LABEL.vocab.itos
    print('label index: ',len(LABEL.vocab.stoi))
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
    print(texicon)
    return TEXT,LABEL,vocab_data,vocab_valid_data,texicon

def build_dataiterator(TEXT,LABEL,vocab_train_data,vocab_valid_data,device,BATCH_SIZE):
    # LABEL = data.Field(sequential=False,tokenize=y_tokenize)
    fields =  [('text', TEXT), ('label', LABEL)]
    vocab_train_data=data.Dataset(vocab_train_data,fields,filter_pred=None)
    vocab_valid_data = data.Dataset(vocab_valid_data, fields, filter_pred=None)

    print('vocab_train: ',len(vocab_train_data))

    vocab_train_iterator,vocab_valid_iterator= data.BucketIterator.splits(
        (vocab_train_data,vocab_valid_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=False,
        sort_key=lambda x: len(x.text),
        device=device)

    return vocab_train_iterator


if __name__=='__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
  
    data_path = r'../data/trec/'
    vocab_file = 'trecknowledgebase.csv'
  
    SEED=1
    TEXT, LABEL, vocab_train_data,vocab_valid_data,texicon = data_loader(data_path, vocab_file,SEED)

    BATCH_SIZE =200

    vocab_train_iterator= build_dataiterator(TEXT, LABEL, vocab_train_data,vocab_valid_data,device,BATCH_SIZE)
    headers = ['text', 'label']
    rows,drows = [],[]
    tokenword=[]
    doubleword, doublewordlabel=[],[]
    for fi in range(len(vocab_train_iterator.dataset.examples)):
        if len(vocab_train_iterator.dataset.examples[fi].text) == 1 and tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0] not in tokenword:
            tokenword.append(tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0])
        else:
            doubleword.append(tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0])
            doublewordlabel.append(vocab_train_iterator.dataset.examples[fi].label)
            print('double',tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0])
    tokenword,double = [],[]
    haveindouble=[]
    for fi in range(len(vocab_train_iterator.dataset.examples)):
        if len(vocab_train_iterator.dataset.examples[fi].text) > 1:
        
            print('more than 1',tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text))
        else:

            if tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0] not in doubleword:
                row = []
                row.append(tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0])

                row.append(vocab_train_iterator.dataset.examples[fi].label)
                rows.append(row)
                tokenword.append(tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0])
            elif tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0] in doubleword and tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0] not in haveindouble:
                haveindouble.append(tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0])
                drow=[]
                drow.append(tokenizer.convert_ids_to_tokens(vocab_train_iterator.dataset.examples[fi].text)[0])
                drow.append(vocab_train_iterator.dataset.examples[fi].label)
                drows.append(drow)

              
    with open('/media/server01/sda2/zixiao/knowledgemapping/textclassification/data/trec/sentidouble.csv', 'w', newline='') as fd:
        f_csv = csv.writer(fd)

        f_csv.writerows(drows)
    exit()
