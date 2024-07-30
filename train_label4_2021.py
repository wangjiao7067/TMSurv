# -*- coding: utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import copy
import torch
import joblib
import random
import json
import math
import sys
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import time as sys_time
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import KFold 
from torch_geometric.data import DataLoader
from ptflops import get_model_complexity_info
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index as ci
from sklearn.model_selection import StratifiedKFold
from tmsurv_label4_2021 import TMSurv
from util import Logger, get_patients_information,get_all_ci,get_val_ci,adjust_learning_rate
from mae_utils import generate_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

def accuracy_cox(y_hat, y):
    correct = np.sum(y_hat == y)
#     print('correct:', correct)
    return correct / len(y)

def prediction(all_data, v_model, val_id, patient_and_time, patient_sur_type, args):
    v_model.eval()
       
    status_all = []
    y_hat = []
    y = []
    
    batch_hazards = None
    batch_S = None
    batch_label = []
    batch_censor = []
    
    val_hazards = {}
    
    iter = 0
    
    with torch.no_grad():
        for i_batch, id in enumerate(val_id):

            graph = all_data[id].to(device)
            label = graph.label

            hazards, S, Y_hat = v_model(graph)
            
            y.append(label.cpu())
            y_hat.append(Y_hat[0][0].cpu())
            val_hazards[id] = torch.min(hazards).cpu().detach().numpy()

            status_all.append(patient_sur_type[id])
            
            batch_label.append(label)
            batch_censor.append(patient_sur_type[id])
            
            if iter == 0 or batch_hazards == None:
                batch_hazards = hazards
                batch_S = S
            else:
                batch_hazards = torch.cat([batch_hazards, hazards])
                batch_S = torch.cat([batch_S, S])

            iter += 1        
            
    status_all = np.asarray(status_all)

    loss_surv = nll_loss(batch_hazards, batch_S, batch_label, status_all)
    loss = loss_surv

    val_ci_ = get_val_ci(val_hazards,patient_and_time,patient_sur_type)
    y_hat = np.asarray(y_hat)
    y = np.asarray(y)
    accuracy = accuracy_cox(y_hat, y)
    return loss.item(), val_ci_, accuracy

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):

    batch_size = len(Y)
    Y = torch.tensor(Y).to(device)
    c = torch.tensor(c).to(device)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    
    S_padded = torch.cat([torch.ones_like(c).to(device), S], 1)
  
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))

    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def train_a_epoch(model, train_data, all_data, patient_and_time, patient_sur_type, batch_size,
                  optimizer, epoch, format_of_coxloss,args):
    model.train() 

    status_all = []
    y_hat = []
    y = []
    
    iter = 0
    loss_nn_all = [] 
    train_hazards = {}
    all_loss = 0.0
    
    batch_hazards = None
    batch_S = []
    batch_label = []
    batch_censor = []
    
    for i_batch, id in enumerate(train_data):
        iter += 1 

        graph = all_data[id].to(device)
        label = graph.label

        hazards, S, Y_hat = model(graph)
        y.append(label.cpu())
        y_hat.append(Y_hat[0][0].cpu())
        
        train_hazards[id] = torch.min(hazards).cpu().detach().numpy()

        status_all.append(patient_sur_type[id])

        batch_label.append(label)
        batch_censor.append(patient_sur_type[id])
        
        if iter == 0 or batch_hazards == None:
            batch_hazards = hazards
            batch_S = S
        else:
            batch_hazards = torch.cat([batch_hazards, hazards])
            batch_S = torch.cat([batch_S, S])

        if iter % batch_size == 0 or i_batch == len(train_data)-1:

            status_all = np.asarray(status_all)

            if np.max(status_all) == 0:
                batch_hazards = None
                batch_S = None
                status_all = []
                iter = 0

                continue

            optimizer.zero_grad() 

            all_loss_surv = nll_loss(batch_hazards, batch_S, batch_label, status_all)
            loss_surv = all_loss_surv
            loss = loss_surv   

            all_loss += loss.item()
            loss.backward()
            if epoch == 0:
                print('*',end='')
            else:  
                optimizer.step()

            torch.cuda.empty_cache()
            batch_hazards = None
            batch_S = None
            status_all = []
            batch_label = []
            batch_censor = []
            loss_nn_all.append(loss.data.item())
            iter = 0               

    all_loss = all_loss/len(train_data)*batch_size
    t_train_ci = get_val_ci(train_hazards,patient_and_time,patient_sur_type)
    y_hat = np.asarray(y_hat)
    y = np.asarray(y)
    accuracy = accuracy_cox(y_hat, y)

    return all_loss,t_train_ci, accuracy


def main(args): 
    start_seed = args.start_seed
    cancer_type = args.cancer_type
    repeat_num = args.repeat_num
    drop_out_ratio = args.drop_out_ratio
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    details = args.details
    fusion_model = args.fusion_model
    format_of_coxloss = args.format_of_coxloss
    if_adjust_lr = args.if_adjust_lr
    print('start')
    label = "{} {} lr_{} {}_coxloss".format(cancer_type, details, lr, format_of_coxloss)

    print(label)
    
    patients = joblib.load('')
    sur_and_time = joblib.load('')
    label = joblib.load('')
    all_data = joblib.load('')

    all_fold_test_ci = []

    repeat = -1
    # start_seed是0，repeat_num是5
    for seed in range(start_seed, start_seed+repeat_num):
        repeat += 1
        setup_seed(0)
            
        seed_patients = []
        
        test_fold_ci = []
        test_fold_acc = []

        val_fold_ci = []
        val_fold_acc = []

        train_fold_ci = []
        n_fold = 0

        test_data_ = []
        train_val_data = list(set(patients) - set(test_data_))
        # kf_label是生存状态
        patient_sur_type_all, patient_and_time_all, kf_label_all = get_patients_information(patients, sur_and_time)
        patient_sur_type, patient_and_time, kf_label = get_patients_information(train_val_data, sur_and_time)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, val_index in kf.split(train_val_data, kf_label):
            fold_patients = []
            n_fold += 1
            print('fold: ', n_fold)
             
            if fusion_model == 'TMSurv':
                model = TMSurv(in_feats=1000, n_hidden=args.n_hidden,
                               out_classes=args.out_classes, dropout=drop_out_ratio).to(device)

            optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            
            train_data = np.array(train_val_data)[train_index]
            val_data = np.array(train_val_data)[val_index]
            test_data = test_data_

            print(len(train_data), len(val_data), len(test_data))
            fold_patients.append(train_data)
            fold_patients.append(val_data)
            fold_patients.append(test_data)
            seed_patients.append(fold_patients)
            
            best_loss = 9999
            best_val_ci = 0
            best_val_acc = 0
            tmp_train_ci = 0
            
            model_info = pd.DataFrame(columns=['epoch', 'train_loss', 'train_ci', 'train_acc', 'val_loss', 'val_ci', 'val_acc'])

            for epoch in range(epochs):
                
                if if_adjust_lr:
                    adjust_learning_rate(optimizer, lr, epoch, lr_step=20, lr_gamma=args.adjust_lr_ratio)
                model.train()
                all_loss, t_train_ci, train_acc = \
                    train_a_epoch(model, train_data, all_data, patient_and_time_all, patient_sur_type_all, batch_size,
                                  optimizer, epoch, format_of_coxloss, args)

                model.eval()
                v_loss, val_ci, val_acc = prediction(all_data, model, val_data,
                                                                    patient_and_time_all, patient_sur_type_all, args)
                if val_ci >= best_val_ci and epoch > 1:
                    best_val_ci = val_ci
                    best_val_acc = val_acc
                    tmp_train_ci = t_train_ci
                    print(val_ci)
#                     t_model = copy.deepcopy(model)
                
                if epoch == epochs-1:
                    t_model = copy.deepcopy(model)

                print("epoch: {:2d}, train_loss: {:.4f}, train_ci: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_ci: {:.4f}, val_acc: {:.4f}".format(epoch, all_loss, t_train_ci, train_acc, v_loss, val_ci, val_acc))
                new_row = {'epoch':epoch+1, 'train_loss':all_loss, 'train_ci':t_train_ci, 'train_acc': train_acc, 'val_loss':v_loss, 'val_ci':val_ci, 'val_acc': val_acc}
                tmp_df = pd.DataFrame(new_row, index=[0])
                model_info = pd.concat([model_info, tmp_df], ignore_index=True)
            save_excel_path = ''
            model_info.to_excel(save_excel_path, index=False)
            

            t_model.eval() 

            t_test_loss, test_ci, test_accuracy = prediction(all_data, t_model, test_data, patient_and_time_all,
                                                       patient_sur_type_all, args)
            test_fold_ci.append(test_ci)
            test_fold_acc.append(test_accuracy)
            val_fold_ci.append(best_val_ci)
            val_fold_acc.append(best_val_acc)
            train_fold_ci.append(tmp_train_ci)

            print('test ci:', test_ci)

            torch.save(t_model.state_dict(), '')
            del model, train_data, t_model

        print('seed: ',seed)
        print('test fold ci:')
        for x in test_fold_ci:
            print(x)
            
        print('test fold acc:')
        for x in test_fold_acc:
            print(x)
        
        print('val fold ci:')
        for x in val_fold_ci:
            print(x)
        
        print('val fold acc:')
        for x in val_fold_acc:
            print(x)
    
        all_fold_test_ci.append(test_fold_ci) 

    print('summary :')
    print(label)  
    
    print('fusion test fold ci')
    for i,x in enumerate(all_fold_test_ci):       
        print(x)

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_type", type=str, default="hecktor", help="Cancer type")
    parser.add_argument("--train_use_type", type=list, default=['ct', 'pt', 'cli'], help='train_use_type,Please keep '
                                                                'the relative order of ct, pt, cli')
    parser.add_argument("--format_of_coxloss", type=str, default="one", help="format_of_coxloss:multi,one")
    parser.add_argument("--start_seed", type=int, default=0, help="start_seed")
    parser.add_argument("--repeat_num", type=int, default=1, help="Number of repetitions of the experiment")
    parser.add_argument("--fusion_model", type=str, default="TMSurv", help="")
    parser.add_argument("--drop_out_ratio", type=float, default=0.5, help="Drop_out_ratio")
    parser.add_argument("--lr", type=float, default=0.00003, help="Learning rate of model training")
    parser.add_argument("--epochs", type=int, default=100, help="Cycle times of model training")
    parser.add_argument("--batch_size", type=int, default=64, help="Data volume of model training once")
    parser.add_argument("--n_hidden", type=int, default=256, help="Model middle dimension")    
    parser.add_argument("--out_classes", type=int, default=256, help="Model out dimension")
    parser.add_argument("--mix", action='store_true', default=True, help="mix mae")
    parser.add_argument("--if_adjust_lr", action='store_true', default=True, help="if_adjust_lr")
    parser.add_argument("--adjust_lr_ratio", type=float, default=0.5, help="adjust_lr_ratio")
    parser.add_argument("--if_fit_split", action='store_true', default=False, help="fixed division/random division")
    parser.add_argument("--details", type=str, default='', help="Experimental details")
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        args = get_params()
        main(args)
    except Exception as exception:
        raise
    
