import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import util
from sklearn.metrics import accuracy_score, f1_score
import json
import numpy as np
import torch.nn.functional as F
import random
from torchtext.LEGACY import data
import datasets

class MLP2(nn.Module):
    # define model elements
    def __init__(self,output_dim,vocab_size=20817,pad_idx=1, embedding_dim=300):
        super(MLP2, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dim =embedding_dim
        # input to first hidden layer
        self.hidden1 = nn.Linear( self.embedding_dim, 512)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(512, self.embedding_dim )
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 =nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(self.embedding_dim , 512)
        nn.init.kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()

        self.hidden4 = nn.Linear(512, 300)
        nn.init.xavier_uniform_(self.hidden4.weight)
        self.act4 = nn.Sigmoid()

        self.fc = nn.Linear(300, output_dim)

    # forward propagate input
    def forward(self, X):  #X[batch,768]
        # input to first hidden layer
        X = self.hidden1(X)  #[768,512]
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X2 = self.act2(X) #[512,768]
        # third hidden layer and output
        X3 = self.hidden3(X2)
        X3 = self.act3(X3)  #[768,512]
        X3= self.hidden4(X3)
        X3 = self.act4(X3)  # [512,300]
        return self.fc(X3),X2



class PreEmbeddings(nn.Module):
    """Construct the embeddings from pretrained embeddings."""

    def __init__(self, config, pretrained_embeddings):
        super().__init__()
        pretrained_embeddings = pretrained_embeddings.astype('float32')
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeddings))
        self.dropout = nn.Dropout(config["embed_dropout_prob"])

    def forward(self, input_ids, class_relatedness_ids=None):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        return embeddings


class RelatedEmbeddings(nn.Module):
    """Construct the embeddings from relatedness between words and labels."""

    def __init__(self, config, related_embeddings):
        super().__init__()
        related_embeddings = related_embeddings.astype('float32')
        self.relatedness = nn.Embedding.from_pretrained(torch.from_numpy(related_embeddings))


    def forward(self, input_ids):
        relatedness = torch.mean(self.relatedness(input_ids), dim=1)
        return relatedness


class LSTMClassifierkef(torch.nn.Module):
    def __init__(self, config, pretrained_embeddings, related_embeddings):
        super().__init__()
        self.config = config
        self.word_embeddings = PreEmbeddings(config, pretrained_embeddings)
        self.relatedness = RelatedEmbeddings(config, related_embeddings)
        self.lstm = nn.LSTM(config["embed_dim"]*2, config["embed_dim"]//2,
                            batch_first=True,
                            bidirectional=True,
                            num_layers=2
                            )
        self.fc1 = nn.Linear(
            config["embed_dim"]//2 + len(config['keywords']) * config['aug'], config["num_classes"])

        self.compat_model1 = MLP2(output_dim=3)
    

    def forward(self, input_ids, modelpath, modeltypelist):
        input_ids = input_ids.permute(1, 0)  #[32,50]
        word_embeddings = self.word_embeddings(input_ids)  #[32,50,300]
       
        self.compat_model1.load_state_dict(torch.load(modelpath))
        _, fine_c_all_hidden = self.compat_model1(word_embeddings)  # [1,768]
        c_all_hidden = torch.cat((word_embeddings, fine_c_all_hidden), dim=2)

        relatedness = self.relatedness(input_ids)  #[32,2]
        lstm_out, (ht, ct) = self.lstm(c_all_hidden.cuda())  #[32,50,300] , ([4,32,150],[4,32,150])
        if self.config["aug"]:
            comb = torch.cat((ht[-1], relatedness), dim=1)  #ht[-1] [32,150]. [32,2]
            x = self.fc1(comb)
        else:
            x = self.fc1(ht[-1])
        return x


def data_processjason(config,data_dir,trainset,testset):
  
    train_data, test_data = [], []
    train_label,test_label=[],[]
    trainfile = open(os.path.join(data_dir, trainset), 'r', encoding='utf-8')
    for trainline in trainfile.readlines():
     
        traindic = json.loads(trainline)
        train_data.append(traindic['text'])
        train_label.append(traindic['label'])
    testfile = open(os.path.join(data_dir, testset), 'r', encoding='utf-8')
    for testline in testfile.readlines():
        testdic = json.loads(testline)
        test_data.append(testdic['text'])
        test_label.append(testdic['label'])
    train_data = datasets.Dataset.from_dict({
        'text': train_data,
        'label': train_label
    })
    test_data = datasets.Dataset.from_dict({
        'text': test_data,
        'label':test_label
    })

  

    vocab2index = util.get_vocab(
        train_data["text"] + test_data["text"], max_size=config["vocab_size"])

    train_data = train_data.map(lambda e: util.encode_sentence(
        e["text"], vocab2index, config))
    train_data.set_format(type='torch', columns=['input_ids', 'label'])
    test_data = test_data.map(lambda e: util.encode_sentence(
        e["text"], vocab2index, config))
    test_data.set_format(type='torch', columns=['input_ids', 'label'])
    train_dl = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    valid_dl = DataLoader(test_data, batch_size=config['batch_size'])

    pretrained_emb = util.load_glove('glove.6B.300d.txt')

    pretrained_embeddings = util.get_emb_matrix(
        pretrained_emb, vocab2index, emb_size=config['embed_dim'])
    keywords_matrix = [pretrained_emb[k] for k in config["keywords"]]
    related_embeddings = util.create_relatedness_matrix(
        keywords_matrix, pretrained_embeddings)

    print(f'embedding matrix shape: {pretrained_embeddings.shape}')
    print(f'relatedness matrix shape: {related_embeddings.shape}')

    return train_dl, valid_dl, pretrained_embeddings, related_embeddings



def train_modeljson(device,model, train_dl,modelpath,modeltypelist,bert_all, config: dict):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config['lr'])
    preds, truth = [], []
    model.train()
    sum_loss = 0.0
    total = 0
    for inputs, targets in train_dl:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        y = targets.to(device) #tensor(32,)
        y_pred = model(modelpath,modeltypelist,bert_all, **inputs)  # three elements: predicts[batch,labelsize], cls_feats:[batch,768], label_feats: [batch,label, 768]

        optimizer.zero_grad()
        loss = F.cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()*y.shape[0]
        total += y.shape[0]
        preds += torch.max(y_pred, 1)[1].detach().cpu()
        truth += y.detach().cpu()
        train_loss, train_acc = sum_loss / total, accuracy_score(truth, preds)
    return train_loss,train_acc


def eval_modeljson(device,model, valid_dl,modelpath,modeltypelist,bert_all):
    model.eval()
    val_sum_loss = 0.0
    val_total = 0
    preds, truth,accs = [], [],[]

    for inputs, targets in valid_dl:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        y = targets.to(device)  # tensor(32,)
        y_hat = model(modelpath,modeltypelist,bert_all, **inputs)  # three elements: predicts[batch,labelsize], cls_feats:[batch,768], label_feats: [batch,label, 768]
        loss = F.cross_entropy(y_hat, y)
        preds += torch.max(y_hat, 1)[1].detach().cpu()
        truth += y.detach().cpu()
        val_total += y.shape[0]
        val_sum_loss += loss.item()*y.shape[0]

    val_loss, val_acc,f1,macro,micro = val_sum_loss / val_total, accuracy_score(truth, preds),f1_score(truth, preds,average='weighted'),f1_score(truth, preds,average='macro'),f1_score(truth, preds,average='micro')

    return val_loss, val_acc,f1,macro,micro

def data_loaderjason(bs, train_file, test_file, path, device):
    text = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True)

    # TEXT = data.Field(tokenize='spacy', include_lengths=True, tokenizer_language='en_core_web_sm')
    label = data.LabelField(dtype=torch.long)
    fields = [('text', text), ('label', label)]
    train_data, test_data = data.TabularDataset.splits(
        path=path,
        train=train_file,
        validation=test_file,
        format='json',
        fields={'text': ('text', text), 'label': ('label', label)},
        skip_header=True)
    train_data, valid_data = train_data.split(random_state=random.seed(1234), split_ratio=0.5)
    MAX_VOCAB_SIZE = 40_00000

    text.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors='glove.6B.300d',
                     unk_init=torch.Tensor.normal_)
    label.build_vocab(train_data)
    texicon = label.vocab.itos
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

def train(model, iterator, optimizer, criterion, modelpath, modeltypelist):
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.Adam(parameters, lr=config['lr'])
    epoch_loss = 0
    # epoch_acc = 0

    model.train()
    preds, truth = [], []
    # count = 0
    for batch in iterator:
        # print('batch',batch)
        optimizer.zero_grad()
        text = batch.text
        # print('t',text.shape)
        predictions = model(text, modelpath, modeltypelist)
        # print('pre:',predictions.size())

        loss = criterion(predictions, batch.label)
        # acc, final = categorical_accuracy(pp_log, batch.label)
        # count += len(batch.label)
        loss.backward()

        optimizer.step()
        preds += torch.max(predictions, 1)[1].detach().cpu()
        truth += batch.label.detach().cpu()

        epoch_loss += loss.item()
        # epoch_acc += acc.item()
    acc=accuracy_score(truth, preds)
    f1, macro, micro = f1_score(truth, preds, average='weighted'), f1_score(truth, preds, average='macro'), f1_score(
        truth, preds, average='micro')
    return epoch_loss / len(iterator), acc,f1, macro, micro


def evaluate(model, iterator, criterion, modelpath, modeltypelist):
    epoch_loss = 0
    # epoch_acc = 0

    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        count = 0
        for batch in iterator:
            text = batch.text
            predictions = model(text, modelpath, modeltypelist)



            loss = criterion(predictions, batch.label)
            # acc, final = categorical_accuracy(prediction_log, batch.label)
            count += len(batch.label)

            preds += torch.max(predictions, 1)[1].detach().cpu()
            truth += batch.label.detach().cpu()

            epoch_loss += loss.item()
            # epoch_acc += acc.item()
    acc = accuracy_score(truth, preds)
    f1, macro, micro = f1_score(truth, preds, average='weighted'), f1_score(truth, preds, average='macro'), f1_score( truth, preds, average='micro')
    return epoch_loss / len(iterator), acc, f1, macro, micro

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    parser = argparse.ArgumentParser(description='Knowledge in Labels Project')
    parser.add_argument('-d', '--data', help='data name', default='imdb',
                        choices=['agnews', 'imdb', 'newsgroup'])
    args = parser.parse_args()
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    modellist = [

        '../mapmodel/glove/noneucentercrossmodelCossim200.pt',
        '../mapmodel/glove/noneucentercrossmodel2.pt',
   

    ]
    modeltypee = [

               
        '3-center_nonneu-centercrossmodelCossim200',
        '3-center_nonneu-centercrossmodel2',
   

    ]
    modeltypelist = [

        'MLP2',
        'MLP2',

    ]

    testsetname = [
         ['CR_Test1.json', 'CR_Test1.json', 'CR_Test1.json', 'CR_Test1.json', 'CR_Test1.json'],
      
    ]
    datasetname = [
        ['cr', 'cr', 'cr', 'cr', 'cr'],
            
    ]

    smallset = [20, 40, 60, 80, 100]
    for set in range(len(smallset)):
        if smallset[set]==100:

            trainsetname= [
               ['CR_Train1.json', 'CR_Train1.json', 'CR_Train1.json', 'CR_Train1.json', 'CR_Train1.json'],
                     
            ]
        
        data_dir = r'../percent/' + str( smallset[set])
        for j in range(len(modellist)):
            modelpath = modellist[j]
            for ds in range(len(trainsetname)):
                testf1, testmacro, testmicro = [], [], []
                bestf1list, bestmalist, bestmilist = [], [], []
                testaccuracy, testaccuracy_l = [], []
                config = settings["lstm"][datasetname[ds][0]]
                config["epochs"] = 50
                config["embed_dropout_prob"] = 0.2
                config["vocab_size"] = None
                config["data_name"] = datasetname[ds][0]
                config["embed_dim"] = 300
                print('lr', config['lr'])
                lr=config['lr']
                testaccuracy, testbestacc = [], []
                testf1, testmacro, testmicro = [], [], []

                for dss in range(len(trainsetname[ds])):
                   
                    train_iterator, valid_iterator, test_iterator, TEXT, LABEL, texicon = data_loaderjason(config["batch_size"], trainsetname[ds][dss], trainsetname[ds][dss], data_dir, device)
                    train_dl, valid_dl, pretrained_embeddings, related_embeddings = data_processjason(config,data_dir,trainsetname[ds][dss],trainsetname[ds][dss])
                    config['aug'] = True
                    model = LSTMClassifierkef(config, pretrained_embeddings, related_embeddings)
                    model = model.to(device)

                    parameters = filter(lambda p: p.requires_grad, model.parameters())
                    optimizer = torch.optim.Adam(parameters, lr= config['lr'])
                    for name, parameter in model.compat_model1.named_parameters():
                        parameter.requires_grad = False
            
                    criterion1 = nn.CrossEntropyLoss()
                    criterion1 = criterion1.to(device)

                    bestl_valid_acc = float(0)
                    bestma, bestmi, bestf1 = float(0), float(0), float(0)
                    bestl_test_acc = float(0)

                    bestl_valid_loss = float('inf')

                    for epoch in range(config['epochs']):
                        train_loss, train_acc,_,_,_ = train(model, train_iterator,optimizer, criterion1 , modelpath,
                                                                                            modeltypelist[j])
                        test_loss, test_acc, f1, macro, micro = evaluate(model, test_iterator,criterion1,  modelpath,
                                                                                            modeltypelist[j])

                        if test_acc >= bestl_test_acc:
                            bestl_test_acc = test_acc
                            bestf1 = f1,
                            bestma = macro
                            bestmi = micro

                        print(f'Epoch: {epoch + 1:02} ')
                        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

                        print(f'\t test. Loss: {test_loss:.3f} |  test. Acc: {test_acc * 100:.2f}%')
              