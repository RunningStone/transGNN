import torch
import pandas as pd
import logging

from histocore.EXP.paras.cohort import CohortParas
from histocore.EXP.paras.dataset import DatasetParas
from torch.utils.data import random_split,TensorDataset,DataLoader
from histocore.DATA.Database.utils import get_weight_sampler

def create_dataset(
                    cohort_para:CohortParas,
                    dataset_para:DatasetParas,
                    target_label_idx:int=0,
                    gene_thresh:float=0.1,
                    ):
    """
    currently only works for GEP matrix graph

    """
    #--------> gene profile matrix r data related
    gcm_Rdata_loc=cohort_para.cohort_data_file#data_para["GCM_data"]
    key=cohort_para.r_key#data_para["GCM_Rdata_key"]#'brca.rnaseq',
    with_transpose = cohort_para.with_transpose#data_para["with_transpose"]
    #--------> gene profile matrix csv/r data related
    label_loc=cohort_para.cohort_file#data_para["task_cohort_file"]#None,
    label_pid = cohort_para.pid_name#data_para["task_PatientID"]#None,
    label_name= cohort_para.targets[target_label_idx]#data_para["task_targets"] #:str="HRD_status",
    label_dict=cohort_para.label_dict#data_para["label_dict"]#:dict={"HRD":0,"HRP":1},
    #--------> data train, test
    split_length=dataset_para.split_ratio#data_para["train_test_ratio"]#:list=[7,3]
    batch_size=dataset_para.batch_size#data_para["batch_size"]#:int=32
    #-----
    from histocore.MODEL.Graph.GAT.utils import get_GeneProfileMatrix
    logging.debug(f"source_loc:{gcm_Rdata_loc},key:{key},label_loc:{label_loc},label_pid:{label_pid}")
    df_data, df_label = get_GeneProfileMatrix(
                                    source_loc=gcm_Rdata_loc,
                                    key=key,
                                    label_loc=label_loc,
                                    p_id=label_pid,
                                    thresh=gene_thresh,
                                    with_transpose=with_transpose,
                                    )
    #--------> load data
    features_table = torch.Tensor(df_data.iloc[:,1:].to_numpy())
    labels_str_np = df_label[label_name].to_numpy()
    labels_str = torch.Tensor([label_dict[labels_str_np[i]] for i in range(labels_str_np.shape[0])])
    logging.info(f"Data table generated with shape {features_table.shape} and {labels_str.shape}")
    # get datasets [train,test,..]
    full_dataset = TensorDataset(features_table, labels_str)
    
    train_size = int(split_length[0] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # split dataset with random or K-fold
    trainset,testset = random_split(full_dataset,
                    lengths=[train_size,test_size],
                    generator=torch.Generator().manual_seed(0)
                    )
    sampler = get_weight_sampler(trainset)

    train_loader = DataLoader(dataset=trainset, 
                                    batch_size=batch_size,sampler=sampler) #shuffle=True)
    test_loader = DataLoader(dataset=testset, 
                                    batch_size=batch_size, shuffle=False)
    
    return trainset,train_loader,testset,test_loader,df_data

def create_Kth_fold(df_data,df_label,kth:int=0):
    pass