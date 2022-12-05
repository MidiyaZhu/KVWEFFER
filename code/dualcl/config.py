import os
import sys
import time
import torch
import random
import logging
import argparse
from datetime import datetime


def get_config():
    parser = argparse.ArgumentParser()
    num_classes = {'sst2': 2, 'subj': 2, 'trec': 6, 'pc': 2, 'cr': 2,'d2': 6, 'em': 4, 'PL': 2,'meld':7,'aman':7,'isear':7,'wos':7}
    ''' Base '''
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='sst2', choices=num_classes.keys())
    parser.add_argument('--model_name', type=str, default='bert', choices=['bert', 'roberta'])
    parser.add_argument('--method', type=str, default='dualcl', choices=['ce', 'scl', 'dualcl'])
    parser.add_argument('--modeltype', type=str, default='MLP2', choices=['MLP2', 'MLP_Secondlayer', 'MLP'])
    parser.add_argument('--modellosstype', type=str, default='noneucentercrossmodel2', choices=[ 'noneucentercrossmodelCossim','noneucentercrossmodel2'])
    ''' Optimization '''
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--decay', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=0.1)
    ''' Environment '''
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.num_classes = num_classes[args.dataset]
    args.device = torch.device(args.device)
    args.log_name = '{}_{}_{}_{}_{}_{}.log'.format(args.dataset, args.model_name, args.method,args.modeltype,args.modellosstype, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args, logger
