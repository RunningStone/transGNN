from transGNN.Data.paras import ReaderParas
import torch
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
    def get_adj_M(self,data_df=None):
        """
        a pre_processing function for create graph etc.
        in:
            data_df: pandas.DataFrame, data table with patientID as index in the first column
        """
        # drop patientID
        df = self.df_data if data_df is None else data_df
        # 重置索引，此时pID变为一列
        #df_reset = df.reset_index()
        df_reset = df
        adj_thresh = self.paras.adj_thresh
        #---->preprocessing steps
        #->get adj matrix
        adj_matrix = WGCNA_py(df_reset)
        #->thresh adj matrix
        adj_M, threshed_M = thresh_adj_matrix(adj_matrix,adj_thresh)
        #->set adj matrix
        self.set_adj_M(threshed_M)
        
    
    def data_readfiles(self):
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
                                        #---> data
                                        data_file_loc=self.paras.data_file_loc,
                                        data_pid=self.paras.data_pid,
                                        key=self.paras.r_file_key,
                                        #---> label
                                        label_file_loc=self.paras.label_file_loc,
                                        label_p_id=self.paras.label_pid,
                                        #---> preprocess
                                        thresh=self.paras.gene_thresh,
                                        with_transpose=self.paras.with_transpose,
                                        )
        self.df_data = df_data

        # df_data include patientID, so drop it
        features_table = torch.Tensor(df_data.iloc[:,1:].to_numpy())
        labels_str_np = df_label[self.paras.label_name].to_numpy()
        labels_str = torch.Tensor([self.paras.label_dict[labels_str_np[i]] for i in range(labels_str_np.shape[0])])
        
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
