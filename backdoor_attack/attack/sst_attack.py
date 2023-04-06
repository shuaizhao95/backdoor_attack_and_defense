import argparse
import os
import sys
import time

curr_path = os.path.dirname(os.path.abspath(__file__))	# 获取当前文件所在目录的绝对路径
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))	# 获取当前文件所在目录的父目录的父目录的绝对路径
print(parent_path)
sys.path.append(parent_path)

import numpy as np
import torch
import torch.nn as nn
from models.model import BERT, LSTM
from data_preprocess.dataset import BERTDataset, bert_fn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

SEED=1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def adjust_lr(optimizer):
    lr = optimizer.param_groups[0]['lr']
    for param_group in optimizer.param_groups:
        adjusted_lr = lr * 0.9
        param_group['lr'] = adjusted_lr if adjusted_lr > 1e-5 else 1e-5
        print("Adjusted learning rate: %.4f" % param_group['lr'])

def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def evaluation(model, device, loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for datapoint in loader:
            padded_text, attention_masks, labels = datapoint
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output, _ = model(padded_text, attention_masks)
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc


def train(model, optimizer, device, epoch, save_path, train_loader_clean, dev_loader_clean, test_loader_clean):
    model.load_state_dict(torch.load(os.path.join(save_path, f"best.ckpt")))
    test_attack_acc = evaluation(model, device, test_loader_clean)
    print('test attack acc: %.4f' % (test_attack_acc))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--data', type=str, default='sst')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.002)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--clean_data_path', type=str, default='data/sst-2/attack/')
    parser.add_argument('--save_path',type=str,default='attack/models/clean_bert_base_tune_sst_mlp0_adam_lr2e-5_bs32_weight0.002/')
    parser.add_argument('--pre_model_path', type=str, default='bert',help='bert-base-uncased')
    parser.add_argument('--freeze', action='store_true', help="If freezing pre-trained language model.")
    parser.add_argument('--mlp_layer_num', default=0, type=int)
    parser.add_argument('--mlp_layer_dim', default=768, type=int)

    args = parser.parse_args()

    lr = args.lr
    data_selected = args.data
    batch_size = args.batch_size
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    epoch = args.epoch
    save_path = args.save_path
    mlp_layer_num = args.mlp_layer_num
    mlp_layer_dim = args.mlp_layer_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.pre_model_path)

    clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.clean_data_path)

    clean_train_dataset, clean_dev_dataset, clean_test_dataset = BERTDataset(
        clean_train_data, tokenizer), BERTDataset(clean_dev_data, tokenizer), BERTDataset(clean_test_data, tokenizer)
    train_loader_clean = DataLoader(clean_train_dataset, shuffle=True, batch_size=batch_size, collate_fn=bert_fn)
    dev_loader_clean = DataLoader(clean_dev_dataset, shuffle=False, batch_size=batch_size, collate_fn=bert_fn)
    test_loader_clean = DataLoader(clean_test_dataset, shuffle=False, batch_size=batch_size, collate_fn=bert_fn)

    class_num = 4 if data_selected=='ag' else 2
    model = BERT(args.pre_model_path, mlp_layer_num,class_num=class_num, hidden_dim=mlp_layer_dim).to(device)

    if args.freeze:
        for param in model.bert.parameters():
            param.requires_grad = False

    if optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    
    sys.stdout.flush()
    train(model, optimizer, device, epoch, save_path, train_loader_clean, dev_loader_clean, test_loader_clean)

if __name__ == '__main__':
    main()
