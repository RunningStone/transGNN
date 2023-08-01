class ReaderParas:

    data_file_loc:str = None, # data file location
    r_file_key:str = None, # key for r data
    label_file_loc:str = None, # label file location
    label_pid:str = None, # patient id for label file

    with_transpose:bool = False, # transpose the data matrix
    label_name:str = None, # label name
    label_dict:dict = None, # label dictionary to map label to int

    """
    parameters for GEP
    """
    gene_thresh:float = 0.1# threshold for gene expression score. gene lower than this value will be removed
    adj_thresh:float = 0.1# threshold for adj matrix. edge lower than this value will be removed(set as 0)

    #--------> data train, test
    test_ratio:float = 0.2# ratio for train and test
    batch_size:int = 32# batch size for train and test