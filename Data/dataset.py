import torch
import pandas as pd
import logging

from transGNN.Data.paras import ReaderParas
from transGNN.Data.preprocessing import DataReader
from torch.utils.data import random_split,TensorDataset,DataLoader



def create_dataset(
        dataset_para:ReaderParas,):
    """
    from data source into dataset
    """
    reader = DataReader(dataset_para) 
    features_table, labels_str = reader.data_readfiles()
    reader.data_preprocessing() # get adj matrix

    full_dataset = TensorDataset(features_table, labels_str)

    test_size = int(dataset_para.test_ratio * len(full_dataset))
    train_size = len(full_dataset) - test_size
    
    # split dataset with random or K-fold
    trainset,testset = random_split(full_dataset,
                    lengths=[train_size,test_size],
                    generator=torch.Generator().manual_seed(0)
                    )
    # 从dataset生成sampler要求每个类别按照一定比例采样
    # 生成sampler

    batch_size = dataset_para.batch_size

    # 统计每个类别的数量
    classes, counts = labels_str.unique(return_counts=True)

    # 计算每个类别的权重
    class_weights = 1. / counts.float()

    # 对每个样本根据其类别赋予权重
    weights = class_weights[labels_str.long()]

    w_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, 
                                                             num_samples=len(weights), 
                                                             replacement=True)
    # 生成dataloader
    train_loader = DataLoader(dataset=trainset, 
                                    batch_size=batch_size,sampler=w_sampler) #shuffle=True)
    test_loader = DataLoader(dataset=testset, 
                                    batch_size=batch_size, shuffle=False)
    
    return train_loader,test_loader,trainset,testset,reader,class_weights