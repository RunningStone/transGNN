class ReaderParas:
    """
    parameters for GEP
    """
    gene_thresh:float = 0.1# threshold for gene expression score. gene lower than this value will be removed
    adj_thresh:float = 0.1# threshold for adj matrix. edge lower than this value will be removed(set as 0)
