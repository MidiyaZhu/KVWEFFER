import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



class MLP2(nn.Module):
    # define model elements
    def __init__(self,output_dim):
        super(MLP2, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.embedding_dim = 768
        # input to first hidden layer
        self.hidden1 = nn.Linear(  self.embedding_dim, 512)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(512, 768)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 =nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(768, 512)
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


class Transformer_withknowledge_CATsentimentatt2(nn.Module):

    def __init__(self, base_model, num_classes, method):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.method = method
        self.linear = nn.Linear(base_model.config.hidden_size*2, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in base_model.parameters():
            param.requires_grad_(True)
        self.compat_model1 = MLP2(3)
       
        self.linear_first = torch.nn.Linear(base_model.config.hidden_size, 100)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(100, 1)
        self.linear_second.bias.data.fill_(0)
        # self.linear_final = torch.nn.Linear(hidden_dim, num_labels)
        self.r = 1

    # self.init_weights()
    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def StructuredSelfAttention(self, outputs):
        x = torch.tanh(self.linear_first(outputs))  # [32,56,100]
        x = self.linear_second(x)  # [32,56,1]
        x = self.softmax(x, 1)  # [32,56,1]
        attention = x.transpose(1, 2)  # [32,1,56]
        sentence_embeddings = attention @ outputs  # [32,1,256]
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r

        # output = self.linear_final(avg_sentence_embeddings)  # [32,6]

        return avg_sentence_embeddings

    def forward(self, modelpath, modeltypelist,bert_all, *args, **kwargs):
        raw_outputs = self.base_model(*args, **kwargs)
        hiddens = raw_outputs.last_hidden_state  # [16,40,768]
        c_all_hidden = torch.zeros(len(hiddens), len(hiddens[0]), len(hiddens[0][0]) * 2)  #[16,57,1536]
        wordhiddens =  torch.zeros(len(hiddens), len(hiddens[0]), len(hiddens[0][0]) )  #[16,57,767]
        
        self.compat_model1.load_state_dict(torch.load(modelpath))
        for batch in range(len(kwargs['input_ids'])):
            c_all_hidden[batch][0] = torch.cat((hiddens[batch][0], hiddens[batch][0]), 0)
            for words in range(1,len(kwargs['input_ids'][batch])):
                if int(kwargs['input_ids'][batch][words]) in bert_all:
                    _, fine_c_all_hidden = self.compat_model1(bert_all[int(kwargs['input_ids'][batch][words])])   #[1,768]
                    c_all_hidden[batch][words] = torch.cat((hiddens[batch][words],fine_c_all_hidden.squeeze(0)), 0)
                    wordhiddens[batch][words] = fine_c_all_hidden
                else:
                    c_all_hidden[batch][words] = torch.cat((hiddens[batch][words], hiddens[batch][words]), 0)


        c_all_hidden=c_all_hidden.to(device)
        cls_feats_knowledge = self.StructuredSelfAttention(wordhiddens[:, 1:].to(device))
      
      
        cls_feats = torch.cat((hiddens[:, 0, :].to(device),cls_feats_knowledge.to(device)),1) # [16,1536]  
        if self.method in ['ce', 'scl']:
            label_feats = None
            predicts = self.linear(self.dropout(cls_feats))
        else:
            label_feats = c_all_hidden[:, 1:self.num_classes+1, :]
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats) #[16,2]
        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }
        return outputs
