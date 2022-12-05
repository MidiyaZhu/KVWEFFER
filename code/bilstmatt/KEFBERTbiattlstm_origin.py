import torch
from torchtext import data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
from numpy import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from pytorch_transformers import *
from transformers import BertTokenizer
import os
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
print('max input length', max_input_length)


# exit()
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens



def data_loadercsv(bs, train_file,valid_file, test_file, path, device):
    text = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx,

                      )
    label = data.LabelField(dtype=torch.long)
    fields = [('text', text), ('label', label)]
    train_data, valid_data, test_data = data.TabularDataset.splits(path=path,
                                                                   train=train_file,
                                                                   validation=valid_file,
                                                                   test=test_file,
                                                                   format='csv',
                                                                   fields=fields,
                                                                   skip_header=True)
    label.build_vocab(train_data)
    texicon = label.vocab.itos
    print('label index: ', len(label.vocab.stoi))
    print(f"Unique tokens in LABEL vocabulary: {len(label.vocab)}")
    print(texicon)
    train_data = data.Dataset(train_data, fields, filter_pred=None)
    valid_data = data.Dataset(valid_data, fields, filter_pred=None)
    test_data = data.Dataset(test_data, fields, filter_pred=None)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=bs,
        sort=False,
        sort_key=lambda x: len(x.text),
        device=device
    )

    return train_iterator, valid_iterator, test_iterator, text, label, texicon


class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        # [b, 784] => [b, 20]
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        # [b, 20] => [b ,784]
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_dim),
            nn.Sigmoid()

        )

    def forward(self, x):
        # batchsz = x.size(0)

        # flatten
        # x = x.view(batchsz, 784)
        # encoder
        x = self.encoder(x.cuda())
        # decoder
        y = self.decoder(x.cuda())

        # reshape
        # x = x.view(batchsz, 1, 28, 28)
        return y.cuda()


bert = BertModel.from_pretrained('bert-base-uncased')


class ClassificationBert_LSTM(nn.Module):
    def __init__(self, num_labels, batch_size, hidden_dim, n_layers, dropout, bidirectional):
        super(ClassificationBert_LSTM, self).__init__()
        # Load pre-trained bert model
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = bert
        self.linear = nn.Linear(hidden_dim, num_labels)

        self.embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.LSTM(self.embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=n_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

        self.dropout = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)

        self.linear_first = torch.nn.Linear(hidden_dim, 100)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(100, 1)
        self.linear_second.bias.data.fill_(0)
        self.linear_final = torch.nn.Linear(hidden_dim, num_labels)

        self.r = 1

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def StructuredSelfAttention(self, outputs, hidden_state, embdim):
        # output [32,56,256]
        # embeddings = self.embeddings(x)
        # outputs, hidden_state = self.lstm(embeddings.view(self.batch_size, self.max_len, -1),
        #                                        self.hidden_state)
        x = F.tanh(self.linear_first(outputs))  # [32,56,100]
        x = self.linear_second(x)  # [32,56,1]
        x = self.softmax(x, 1)  # [32,56,1]
        attention = x.transpose(1, 2)  # [32,1,56]
        sentence_embeddings = attention @ outputs  # [32,1,256]
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r

        output = self.linear_final(avg_sentence_embeddings)  # [32,6]

        return output, x

    def forward(self, x):
        with torch.no_grad():
            all_hidden, pooler = self.bert(x.cuda())

        embdim = all_hidden.shape[0]
        pooled_output, (hidden, cell) = self.lstm(all_hidden)
        attn_output, word_weights = self.StructuredSelfAttention(pooled_output, hidden, embdim)

        return attn_output, word_weights.squeeze(2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    a = correct.sum()
    b = torch.FloatTensor([y.shape[0]]).cuda()
    c = max_preds
    return correct.sum().cuda(), max_preds


parm = {}


def parm_to_excel(excel_name, key_name, parm):
    with pd.ExcelWriter(excel_name) as writer:
        [output_num, input_num, filter_size, _] = parm[key_name].size()
        for i in range(output_num):
            for j in range(input_num):
                data = pd.DataFrame(parm[key_name][i, j, :, :].detach().numpy())
                # print(data)
                data.to_excel(writer, index=False, header=True, startrow=i * (filter_size + 1),
                              startcol=j * filter_size)



def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    count = 0
    for batch in iterator:
        # print('batch',batch)
        optimizer.zero_grad()
        text = batch.text
        # print('t',text.shape)
        predictions, cat = model(text)
        # print('pre:',predictions.size())
        pp_log = predictions

        loss = criterion(pp_log, batch.label)
        acc, final = categorical_accuracy(pp_log, batch.label)
        count += len(batch.label)
        loss.backward()

        optimizer.step()
     
          
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / count

def evaluate(model, iterator, criterion, texicon):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    preds, truth= [], []
    with torch.no_grad():
        count = 0
        for batch in iterator:
            text = batch.text
            predictions, cat = model(text)

            prediction_log = predictions

            loss = criterion(prediction_log, batch.label)
            acc, final = categorical_accuracy(prediction_log, batch.label)
            count += len(batch.label)
            preds += torch.max(predictions, 1)[1].detach().cpu()
            truth += batch.label.detach().cpu()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    f1, macro, micro = f1_score(truth, preds, average='weighted'), f1_score(truth, preds, average='macro'), f1_score(truth, preds, average='micro')
    return epoch_loss / len(iterator), epoch_acc / count,f1, macro, micro
        # conf_matrix, true_matrix, portion_matrix , P, TP, FN

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


#

if __name__ == '__main__':
    version =0
    dataset_name = 'bilstmvaild'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    smallset = [
    100


    ]
    trainset = [['aman_train.csv','aman_train.csv','aman_train.csv','aman_train.csv','aman_train.csv'],
                        ['isear_train.csv', 'isear_train.csv', 'isear_train.csv', 'isear_train.csv',  'isear_train.csv'],]

    testset = [
        ['aman_test.csv', 'aman_test.csv', 'aman_test.csv', 'aman_test.csv', 'aman_test.csv'],
        ['isear_test.csv', 'isear_test.csv', 'isear_test.csv', 'isear_test.csv', 'isear_test.csv'],
    ]
    datasetname = [
        ['aman', 'aman', 'aman', 'aman', 'aman'],
        ['isear', 'isear', 'isear', 'isear', 'isear'],
    ]


    for set in tqdm(range(len(smallset)), ascii=True, desc='smallset'):

        for ds in tqdm(range(len(trainset)), ascii=True, desc='set'):
      
            for dss in tqdm(range(len(trainset[ds])), ascii=True, desc='dataset'):


                data_path = r'../testdata/percent/' + str(smallset[set])

                train_file = trainset[ds][dss]
                vaild_file = vaildset[ds][dss]
                test_file = testset[ds][dss]

                BATCH_SIZE = 32

                train_iterator, valid_iterator, test_iterator, TEXT, LABEL, texicon = data_loadercsv(
                    BATCH_SIZE,
                    train_file, vaild_file,
                    test_file, data_path,
                    device)


                N_EPOCHS = 20
                lr = 9e-4
                OUTPUT_DIM = len(LABEL.vocab)
                DROPOUT = 0.5
                N_LAYERS = 2
                BIDIRECTIONAL = True

                model = ClassificationBert_LSTM(num_labels=OUTPUT_DIM, batch_size=BATCH_SIZE,
                                                hidden_dim=256,
                                                n_layers=N_LAYERS,
                                                dropout=DROPOUT, bidirectional=BIDIRECTIONAL)

                for name, param in model.named_parameters():
                    if name.startswith('bert'):
                        param.requires_grad = False
              
                torch.manual_seed(1)
                torch.backends.cudnn.deterministic = True

                pathl = r'LSTM-VALLOSS-np-' + str(version) + dataset_name + 'model.pt'

                print(f'The model has {count_parameters(model):,} trainable parameters')

                optimizer = optim.Adam(model.parameters(), lr=lr)

                criterion1 = nn.CrossEntropyLoss()


                model = model.to(device)

                criterion1 = criterion1.to(device)
       
                bestl_valid_acc = float(0)
                bestl_test_acc = float(0)       
                bestl_valid_loss = float('inf')
                
                for epoch in range(N_EPOCHS):

                    start_time = time.time()

                    train_loss, train_acc= train(model, train_iterator,
                                                                                    optimizer, criterion1,
                                                                             )
                    valid_loss, valid_acc, _, _, _ = evaluate(model, valid_iterator, criterion1, texicon)
                  
                    end_time = time.time()

                    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                    if valid_acc >= bestl_valid_acc:
                        bestl_valid_acc = valid_acc
                        bestl_epoch = epoch

                        torch.save(model.state_dict(), pathl)

                    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
                  
                model.load_state_dict(torch.load(pathl))

                test_loss, test_acc,f1,macro,micro = evaluate( model, test_iterator, criterion1, texicon)

                
                print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

          