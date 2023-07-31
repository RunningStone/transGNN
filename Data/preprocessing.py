from transGNN.Data.paras import ReaderParas

class DataReader:
    def __init__(self,reader_paras:ReaderParas):
        self.paras = reader_paras

    def set_adj_M(self,M):
        # set adjacent matrix
        self.M = M#torch.from_numpy(M)
        #self.M.requires_grad = False
    ############################################################################
    # .  some utils functions for use the model
    ############################################################################
    def data_preprocessing(self,data_pack:dict):
        """
        a pre_processing function for create graph etc.
        """
        df_data = data_pack["df"]
        adj_thresh = self.data_paras.adj_thresh
        #---->preprocessing steps
        #->get adj matrix
        adj_matrix = WGCNA_py(df_data)
        #->thresh adj matrix
        adj_M, threshed_M = thresh_adj_matrix(adj_matrix,adj_thresh)
        #->set adj matrix
        self.set_adj_M(threshed_M)
        
    
    def data_readfiles(self,data_file_loc,r_file_key,
                            label_file_loc,label_pid,
                            gene_thresh,with_transpose,
                            label_name,label_dict
                            ):
        """
        a function for read r files as data source
        in:
            data_loc: str, data file location normally is a Rdata file
            r_file_key: str, the key of the data table in the Rdata file based on pyreadr
            label_file_loc: str, label file location
            label_pid: str, the key of the patientID column in the label file
            gene_thresh: float, the threshold of the gene expression value
            with_transpose: bool, if the data is transposed
            label_name: str, the key of the label column in the label file
            label_dict: dict, the dict of the label to get int or one hot label
        out:
            features_table: torch.tensor, the feature table
            labels_str: torch.tensor, the label table
        """
        from transGNN.Data.utils import get_GeneProfileMatrix
        df_data, df_label = get_GeneProfileMatrix(
                                        data_file_loc=data_file_loc,
                                        key=r_file_key,
                                        label_file_loc=label_file_loc,
                                        p_id=label_pid,
                                        thresh=gene_thresh,
                                        with_transpose=with_transpose,
                                        )
        self.df_data = df_data
        features_table = torch.Tensor(df_data.iloc[:,1:].to_numpy())
        labels_str_np = df_label[label_name].to_numpy()
        labels_str = torch.Tensor([label_dict[labels_str_np[i]] for i in range(labels_str_np.shape[0])])
        
        return features_table,labels_str
    



##########################################################################################
# .  some utils functions for use the model
##########################################################################################
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
