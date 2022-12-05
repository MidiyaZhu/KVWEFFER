import os
import json
import torch
import random
from functools import partial
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self, raw_data, label_dict, tokenizer, model_name, method):
        dataset = list()
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            label_id = label_dict[data['label']]
            dataset.append((tokens, label_id))
        self._dataset = dataset
        self._method = method
        self.sep_token = ['[SEP]'] if model_name == 'bert' else ['</s>']
        self._label_list = list(label_dict.keys())
        self._num_classes = len(label_dict)

    def __getitem__(self, index):
        tokens, label_id = self._dataset[index]
        if self._method not in ['ce', 'scl']:
            rand_idx = [i for i in range(self._num_classes)]
            random.shuffle(rand_idx)
            label_list = [self._label_list[i] for i in rand_idx]
            tokens = label_list + self.sep_token + tokens
            label_id = rand_idx[label_id]
        return tokens, label_id

    def __len__(self):
        return len(self._dataset)


def my_collate(batch, tokenizer):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    return text_ids, torch.tensor(label_ids)

# def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name, method, workers):
def load_data(dataset, trainset, testset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name,  method, workers):
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'trec':
        train_data = json.load(open(os.path.join(data_dir, 'TREC_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'TREC_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'description': 0, 'entity': 1, 'abbreviation': 2, 'human': 3, 'location': 4, 'numeric': 5}
    elif dataset == 'cr':
        train_data = json.load(open(os.path.join(data_dir, 'CR_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'CR_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'subj':
        train_data = json.load(open(os.path.join(data_dir, 'SUBJ_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SUBJ_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'subjective': 0, 'objective': 1}
    elif dataset == 'pc':
        train_data = json.load(open(os.path.join(data_dir, 'procon_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'procon_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}

    else:
        raise ValueError('unknown dataset')
    trainset = MyDataset(train_data, label_dict, tokenizer, model_name, method)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name, method)
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers,
                                  collate_fn=partial(my_collate, tokenizer=tokenizer), pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers,
                                 collate_fn=partial(my_collate, tokenizer=tokenizer), pin_memory=True)
    return train_dataloader, test_dataloader


def load_datasmallset(dataset, trainset, testset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name,  method, workers):

    train_data,test_data=[],[]
    trainfile =open(os.path.join(data_dir, trainset), 'r', encoding='utf-8')
    for trainline in trainfile.readlines():
        # if trainline!='{\n':
        traindic = json.loads(trainline)
        train_data.append(traindic)
    testfile = open(os.path.join(data_dir, testset), 'r', encoding='utf-8')
    for testline in testfile.readlines():
        testdic = json.loads(testline)
        test_data.append(testdic)
    if dataset == 'sst2'or dataset == 'cr' or dataset == 'pc' or dataset == 'PL':
        label_dict = {'positive': 1, 'negative': 0}
    elif dataset == 'trec':
        label_dict = {'description': 0, 'entity': 1, 'abbreviation': 2, 'human': 3, 'location': 4, 'numeric': 5}
    elif dataset == 'cr':
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'pc':
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset=='aman':
        label_dict = {"neutral": 0, "disgust": 1, "happy": 2, "surprise": 3, "sad": 4, "angry": 5, "fear": 6}
    elif dataset=='meld':
        label_dict = {"joy": 0, "fear": 1, "anger": 2, "sadness": 3, "neutral": 4, "surprise": 5, "disgust": 6}
    elif dataset=='isear':
        label_dict = {"joy": 0, "fear": 1, "anger": 2, "sadness": 3, "disgust": 4, "shame": 5, "guilt": 6}
    elif dataset=='em':
        label_dict = {"angry": 0, "fear": 1, "happy": 2, "sad": 3}
    elif dataset=='d2':
        label_dict = {"happy": 0, "fear": 1, "angry": 2, "sad": 3, "love": 4, "surprise": 5}
    elif dataset == 'PL':
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'wos':
        label_dict = {'CS': 0, 'Medical': 1,"Civil": 2, "ECE": 3, "biochemistry": 4, "MAE": 5, "Psychology": 6}
    else:
        raise ValueError('unknown dataset')
    trainset = MyDataset(train_data, label_dict, tokenizer, model_name, method)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name, method)
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers,
                                  collate_fn=partial(my_collate, tokenizer=tokenizer), pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers,
                                 collate_fn=partial(my_collate, tokenizer=tokenizer), pin_memory=True)
    return train_dataloader, test_dataloader
