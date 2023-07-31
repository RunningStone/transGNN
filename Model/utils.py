"""
utils functions for GAT method
"""
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import logging
from histocore.DATA.FileIO.table_backends import tableWorker


########################################################################
# .    from file to data source
########################################################################

def get_GeneProfileMatrix(source_loc,
                                key='brca.rnaseq',
                                label_loc=None,p_id=None,
                                thresh=1.0,
                                with_transpose:bool=False,
                                ):
    tw = tableWorker(source_loc)
    tw.read_table(key=key)
    # get patient id and drop score lower than thrh = 1.0
    df2 = tw.df
    # DEBUG:
    df2 = df2.set_index("Unnamed: 0")
    logging.debug(f"{df2[:5]}")
    logging.debug(f"item example: {df2[df2.columns.tolist()[0]]},thresh:{thresh}")
    for name in df2.columns.tolist():
        df2 = df2.drop(df2[df2[name]<thresh].index)
    if with_transpose:
        #  transpose to make  [patient x gene]
        df3 = df2.transpose()
    else:
        df3 = df2
    df3.insert(0,"sample_id",df3.index.to_list())
    df3=df3.reset_index(drop=True)

    if label_loc is not None:
        # add patient id for data
        df3["PatientID"] = [name[:12] for name in df3["sample_id"].to_list()]
        # read label file
        tw_dnad = tableWorker(label_loc)
        tw_dnad.read_table()
        # add patient id for label
        df_dnad = tw_dnad.df
        df_dnad["PatientID"] = [name[:12] for name in df_dnad[p_id].to_list()]
        # merge to confirm all have label
        df_all = pd.merge(df3,df_dnad,on=["PatientID"])

        # get data
        print(df_all.columns)
        df_data = df_all.loc[:,df3.columns.to_list()]
        df_data = df_data.drop(["PatientID"],axis=1)
        # get label
        df_label = df_all.loc[:,df_dnad.columns.to_list()]
    else:
        df_data = df3
        df_label = None
    #print(df3.head(5))
    return df_data, df_label
########################################################################
# .    model requires pre-processing step functions
########################################################################
def WGCNA_py(df_data):
    """
    a small wrapper funciton for pyWGCNA
    """
    import PyWGCNA
    worker = PyWGCNA.WGCNA(name='GPM', 
                        species='human',
                        geneExp=df_data,
                        #geneExpPath=data_loc2, 
                        save=False)
    try:
        worker.preprocess()
        worker.findModules()
    except Exception as e: print(e)
    finally:
        adj_M = worker.adjacency
    return adj_M

def thresh_adj_matrix(adj_M,adj_thresh=None):
    """
    calculate the threshold for adj matrix if adj_thresh is not None
    """
    M = adj_M.copy()
    
    if adj_thresh is not None:
        #M[adj_M>adj_thresh]=1
        M[adj_M<=adj_thresh]=0
    return adj_M, M


########################################################################
# .    model requires gradient calculation functions for XAI
########################################################################
def calc_GAT_gradient(data,model,class_nb=2,concat_nb=3):
    """
    in:
        data: pyg_tensor with x and y
        model: pytorch model
    
    out:
        gradient: saved gradient for y_c = torch.sum(one_hot_labels*out)
        feature_importance: feature importance for current data 
    """
    grad_labels = data.y.to(torch.int64)

    result_dict=model(data)

    GAT_features = result_dict["GAT_features"]
    out = result_dict["logits"]

    # clean gradient
    GAT_features.grad = None
    GAT_features.retain_grad()
    
    # to one hot
    one_hot_labels = F.one_hot(grad_labels, num_classes=class_nb)
    one_hot_labels = one_hot_labels.to(GAT_features.device)
    y_c = torch.sum(one_hot_labels*out)

    y_c.backward(retain_graph=True)
    np_GAT_f_grad = GAT_features.grad.detach().cpu().numpy()
    np_GAT_f_grad = np_GAT_f_grad.reshape([1,int(np_GAT_f_grad.shape[0]/concat_nb),concat_nb])
    #print(np_GAT_f_grad.shape)
    gradients = np.maximum(np_GAT_f_grad, 0)# (1, feature_size,concat_nb)
    feature_importance = np.mean(gradients, 0)

    return gradients,feature_importance

def class_importance(target_class,labels,gradients,concat_nb,feature_dim):
    """
    in:
        target_class:int: target class for importance
        labels:np.ndarray: a list of number as labels
        gradients:
    return:
        class_i_importance:np.ndarray: importance score for class i
    """
    class_i_index = np.argwhere(labels == target_class)
    class_i_gradients = gradients[class_i_index[:, 0], :]
    class_i_importance = np.mean(class_i_gradients, axis=0)
    class_i_importance = np.reshape(class_i_importance, (concat_nb, feature_dim)).T
    from sklearn.preprocessing import normalize
    #print(class_i_importance.shape)
    norm_class_i_importance = normalize(class_i_importance, axis=0, norm='max')
    #print(norm_class_i_importance.shape)
    class_i_importance = np.expand_dims(norm_class_i_importance, axis=0)

    return class_i_importance #[1, feature_dim, concat_nb]
