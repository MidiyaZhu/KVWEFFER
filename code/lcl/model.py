import pandas as pd
import numpy as np
import json
import random
import pickle
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import ElectraForSequenceClassification,ElectraModel,BertModel,BertForSequenceClassification
from util import load_model


class primary_encoder(nn.Module):

    def __init__(self,batch_size,hidden_size,emotion_size,encoder_type):
        super(primary_encoder, self).__init__()
        self.encoder_type=encoder_type
        if encoder_type == "electra":
            options_name = "google/electra-base-discriminator"
            self.encoder_supcon = ElectraModel.from_pretrained(options_name,num_labels=emotion_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing=True

        if encoder_type == "bert":
            self.encoder_supcon = BertModel.from_pretrained('bert-base-uncased',num_labels=emotion_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing = True


        self.pooler_fc = nn.Linear(hidden_size,hidden_size)
        self.pooler_dropout = nn.Dropout(0.1)
        self.label = nn.Linear(hidden_size,emotion_size)

    def get_emedding(self, features):
        x = features[:, 0, :]
        x = self.pooler_fc(x)
        x = self.pooler_dropout(x)
        x = F.relu(x)
        return x


    def forward(self, text,attn_mask):


        if self.encoder_type == "electra":
            supcon_fea = self.encoder_supcon(text,attn_mask,output_hidden_states=True,output_attentions=True,return_dict=True)
            # print('supcon_fea.hidden_states',supcon_fea.hidden_states[-1].shape)
            supcon_fea_cls_logits =  self.get_emedding(supcon_fea.hidden_states[-1])  #supcon_fea.hidden_states[-1]【20,74,768】  supcon_fea_cls_logits【20,768】
            supcon_fea_cls_logits = self.pooler_dropout(self.label(supcon_fea_cls_logits))  # self.label(supcon_fea_cls_logits)【20,7】

            supcon_fea_cls = F.normalize(supcon_fea.hidden_states[-1][:, 0, :], dim=1)  # supcon_fea.hidden_states[-1][:,0,:]【20,768】

        elif self.encoder_type=='bert':
            # print('text3:', text.shape)
            supcon_fea = self.encoder_supcon(text,attn_mask,output_hidden_states=True,output_attentions=True,return_dict=True)
            # print('supcon_fea.hidden_states', supcon_fea.hidden_states[-1].shape)
            supcon_fea_cls_logits = self.get_emedding(supcon_fea.last_hidden_state)  # supcon_fea.hidden_states[-1]【20,74,768】  supcon_fea_cls_logits【20,768】
            supcon_fea_cls_logits = self.pooler_dropout(self.label(supcon_fea_cls_logits)) #self.label(supcon_fea_cls_logits)【20,7】

            supcon_fea_cls = F.normalize(supcon_fea.last_hidden_state[:,0,:],dim=1) #supcon_fea.hidden_states[-1][:,0,:]【20,768】


        return supcon_fea_cls_logits,supcon_fea_cls  #[batch*2,num_class],[batch*2,768]


class weighting_network(nn.Module):

    def __init__(self,batch_size,hidden_size,emotion_size,encoder_type):
        super(weighting_network, self).__init__()
        self.encoder_type = encoder_type
        if encoder_type == "electra":
            options_name = "google/electra-base-discriminator"
            self.encoder_supcon_2 = ElectraForSequenceClassification.from_pretrained(options_name,num_labels=emotion_size)

            ## to make it faster
            self.encoder_supcon_2.electra.encoder.config.gradient_checkpointing=True

        elif encoder_type == "bert":

            self.encoder_supcon_2 = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=emotion_size)

            ## to make it faster
            self.encoder_supcon_2.bert.encoder.config.gradient_checkpointing=True


    def forward(self, text,attn_mask):
        # if text.shape[1]>512:
        #     text=torch.cat((text[:,0:511],text[:,-1].unsqueeze(1)),1)
        #     attn_mask=attn_mask[:,0:512]

        supcon_fea_2 = self.encoder_supcon_2(text,attn_mask,output_hidden_states=True,return_dict=True)

        return supcon_fea_2.logits  #[20,7]




class primary_encoderwithksentimentatt2(nn.Module):

    def __init__(self,batch_size,hidden_size,emotion_size,encoder_type):
        super(primary_encoderwithksentimentatt2, self).__init__()
        self.encoder_type=encoder_type
        if encoder_type == "electra":
            options_name = "google/electra-base-discriminator"
            self.encoder_supcon = ElectraModel.from_pretrained(options_name,num_labels=emotion_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing=True

        if encoder_type == "bert":
            self.encoder_supcon = BertModel.from_pretrained('bert-base-uncased',num_labels=emotion_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing = True

        self.pooler_fc1 = nn.Linear(hidden_size, hidden_size)
        self.pooler_fc2 = nn.Linear(hidden_size,hidden_size)
        self.pooler_dropout = nn.Dropout(0.1)
        self.label = nn.Linear(hidden_size*2,emotion_size)
        self.compat_model1 = MLP2(3)
 

        self.linear_first = torch.nn.Linear(768, 100)
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

    def get_emeddingknowledge(self, x):
        # x = features[:, 0, :] #[64,1536]
        x = self.pooler_fc1(x)
        # x = self.pooler_dropout(x)
        x = F.relu(x)
        return x
    def get_emedding2(self, features):
        x = features[:, 0, :] #[64,1536]
        x = self.pooler_fc2(x)
        # x = self.pooler_dropout(x)
        x = F.relu(x)
        return x


    def forward(self, text,attn_mask,modeltypelist,modelpath,bert_all):

        #text [batch*2,74]
        if self.encoder_type == "electra":
            supcon_fea = self.encoder_supcon(text,attn_mask,output_hidden_states=True,output_attentions=True,return_dict=True)

            supcon_fea_cls_logits =  self.get_emedding(supcon_fea.hidden_states[-1])  #supcon_fea.hidden_states[-1]【20,74,768】  supcon_fea_cls_logits【20,768】
            supcon_fea_cls_logits = self.pooler_dropout(self.label(supcon_fea_cls_logits))  # self.label(supcon_fea_cls_logits)【20,7】

            supcon_fea_cls = F.normalize(supcon_fea.hidden_states[-1][:, 0, :], dim=1)  # supcon_fea.hidden_states[-1][:,0,:]【20,768】

        elif self.encoder_type=='bert':

            supcon_fea = self.encoder_supcon(text,attn_mask,output_hidden_states=True,output_attentions=True,return_dict=True)
        
            hiddens=torch.zeros(len(supcon_fea.last_hidden_state), len(supcon_fea.last_hidden_state[0]), len(supcon_fea.last_hidden_state[0][0]))

            self.compat_model1.load_state_dict(torch.load(modelpath))
            for batch in range(len(text)):
                for index in range(1,len(text[batch])):
                    if int(text[batch][index]) in bert_all:
                        _, fine_c_all_hidden = self.compat_model1(bert_all[int(text[batch][index])])  # [1,768]
                        hiddens[batch][index] = fine_c_all_hidden
            cls_feats_knowledge = self.StructuredSelfAttention(hiddens[:, 1:].cuda())
           
            supcon_fea_cls_logits_knowledge = self.get_emeddingknowledge(cls_feats_knowledge)  #[64,768]
            supcon_fea_cls_logits_lcl = self.get_emedding2(supcon_fea.hidden_states[-1])  # supcon_fea.hidden_states[-1]【20,74,768】  supcon_fea_cls_logits【64,768】
            pooled_output1=torch.cat((supcon_fea_cls_logits_lcl,supcon_fea_cls_logits_knowledge),1)  #[64,1536]
            supcon_fea_cls_logits = self.pooler_dropout(self.label(pooled_output1))  # self.label(supcon_fea_cls_logits)【20,7】


            supcon_fea_cls = torch.cat((F.normalize(supcon_fea.last_hidden_state[:,0,:], dim=1),F.normalize(cls_feats_knowledge, dim=1)),1)  # supcon_fea.hidden_states[-1][:,0,:]【20,768】  this is [CLS] ID 101 's encoding which is not modified

        return supcon_fea_cls_logits,supcon_fea_cls  #[batch*2,num_class],[batch*2,768*2]


class MLP2(nn.Module):
    # define model elements
    def __init__(self, output_dim):
        super(MLP2, self).__init__()
        self.embedding_dim = 768
        # input to first hidden layer
        self.hidden1 = nn.Linear(self.embedding_dim, 512)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(512, 768)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(768, 512)
        nn.init.kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()

        self.hidden4 = nn.Linear(512, 300)
        nn.init.xavier_uniform_(self.hidden4.weight)
        self.act4 = nn.Sigmoid()

        self.fc = nn.Linear(300, output_dim)

    # forward propagate input
    def forward(self, X):  # X[batch,768]
        # input to first hidden layer
        X = self.hidden1(X)  # [768,512]
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X2 = self.act2(X)  # [512,768]
        # third hidden layer and output
        X3 = self.hidden3(X2)
        X3 = self.act3(X3)  # [768,512]
        X3 = self.hidden4(X3)
        X3 = self.act4(X3)  # [512,300]
        return self.fc(X3), X2