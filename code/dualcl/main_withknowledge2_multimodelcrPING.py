import torch
from tqdm import tqdm
from model import Transformer_withknowledge_CATsentimentatt2
from config import get_config
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_datasmallset
from transformers import logging, AutoTokenizer, AutoModel
from numpy import *
from sklearn.metrics import f1_score



class Instructor:

    def __init__(self, args, logger,modelpath,filepath,modeltypelist,bert_all):
        self.args = args
        self.logger = logger
        self.modelpath=modelpath
        self.filepath=filepath
        self.modeltypelist=modeltypelist
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
           
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            base_model = AutoModel.from_pretrained('roberta-base')
        else:
            raise ValueError('unknown model')
        self.model = Transformer_withknowledge_CATsentimentatt2(base_model, args.num_classes, args.method)
        self.model.to(args.device)
        for name, parameter in self.model.compat_model1.named_parameters():
            parameter.requires_grad = False
        

        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

        self.bert_all=bert_all

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0

        self.model.train()
        for inputs, targets in dataloader:
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            outputs = self.model(self.modelpath,self.modeltypelist,self.bert_all,**inputs) #three elements: predicts[batch,labelsize], cls_feats:[batch,768], label_feats: [batch,label, 768]
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
            n_train += targets.size(0)



        return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        preds, truth = [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(self.modelpath,self.modeltypelist,self.bert_all,**inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
                n_test += targets.size(0)
                preds += torch.max(outputs['predicts'], 1)[1].detach().cpu()
                truth += targets.detach().cpu()

            f1, macro, micro = f1_score(truth, preds, average='weighted'), f1_score(truth, preds, average='macro'), f1_score(truth,  preds, average='micro')
        return test_loss / n_test, n_correct / n_test, f1, macro, micro

    def run(self,dataset,data_dir,trainset,testset):
        train_dataloader, test_dataloader = load_datasmallset(dataset=dataset,
                                                      trainset=trainset,
                                                      testset=testset,
                                                      data_dir=data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      model_name=self.args.model_name,
                                                      method=self.args.method,
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
        else:
            raise ValueError('unknown method')
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)
        best_loss, best_acc,best_f1,best_macro,best_micro  = 0, 0,0,0,0

        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc, f1, macro, micro = self._test(test_dataloader, criterion)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss,best_f1,best_macro,best_micro = test_acc, test_loss, f1, macro, micro
            self.logger.info('{}/{} - {:.2f}%'.format(epoch+1, self.args.num_epoch, 100*(epoch+1)/self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc*100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc*100))
           
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc*100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
     
        return best_acc,best_f1,best_macro,best_micro

if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    bert_all = torch.load(
        r'../mappingmodel/Bert/WordListEmbedding.pt')
    for key, value in bert_all.items():
        bert_all[key] = bert_all[key].to(args.device)

    print(args.device)
    modelpath = [
       
        '../mappingmodel/Bert/noneucentercrossmodelCossim200.pt',
        '../mappingmodel/Bert/noneucentercrossmodel2.pt',

    ]
    modellist = [
     
        '3-center_nonneu-centercrossmodelCossim200',
        '3-center_nonneu-centercrossmodel2',

    ]
    modeltypelist = [
      
        'MLP2',
        'MLP2',

    ]
    smallset = [
     # 20,
     #    40,
     #    60,
     #    80,
        100
    ]
    trainset = [

        ['CR_Train1_0.json', 'CR_Train1_1.json', 'CR_Train1_2.json', 'CR_Train1_3.json', 'CR_Train1_4.json'],
     

    ]
    testset = [

        ['CR_Test1.json', 'CR_Test1.json', 'CR_Test1.json', 'CR_Test1.json', 'CR_Test1.json'],
      
    ]

    dataset = [
        ['cr', 'cr', 'cr', 'cr', 'cr'],
    
    ]

    num_classes = {'sst2': 2, 'subj': 2, 'trec': 6, 'pc': 2, 'cr': 2, 'd2': 6, 'em': 4, 'PL': 2, 'meld': 7, 'aman': 7,'isear': 7}
    args.num_epoch = 15

    args.train_batch_size = 32
    for set in range(len(smallset)):
        if smallset[set]==100:
         
            trainset = [
                ['CR_Train1.json', 'CR_Train1.json', 'CR_Train1.json', 'CR_Train1.json', 'CR_Train1.json'],
              
            ]

        for i in range(len(modellist)):
            args.modeltype = modeltypelist[i]
            args.modellosstype = modellist[i]
            for ds in tqdm(range(len(trainset)), ascii=True, desc='set'):
                testaccuracy,bestf1list,bestmalist,bestmilist=[],[],[],[]
                for dss in tqdm(range(len(trainset[ds])), ascii=True, desc='dataset'):
                    args.dataset = dataset[ds][dss]
                    args.num_classes = num_classes[dataset[ds][dss]]
                   
                    data_path = r'../data/testdata/percent/' + str( smallset[set])
                    ins = Instructor(args, logger,modelpath[i],filepath,modeltypelist[i],bert_all)
                    test_acc,test_f1,test_macro,test_micro=ins.run(dataset[ds][dss],data_path,trainset[ds][dss],testset[ds][dss])
                   