import attr 

@attr.s
class GATParas:
    """
    parameters for GAT
    """
    #---->model paras gnn
    input_dim:int = None # input dimension can be any
    nhids:list = [8, 16, 12] # hidden dimension
    nheads:list = [4, 3, 4] # number of heads
    GAT_dropout:float = 0.2 # dropout rate
    norm_nb_nodes: int = 10
    with_layer_norm: bool = True
    which_layer: str = "all"

    #---->model paras mlp
    lin_input_dim:int=720  #the feature extracted by GAT layers
    fc_dim:list = [64, 48, 32]
    fc_dropout:float=0.2
    #-----> paras for classificer
    omic_dim:int=32
    class_nb:int=2
    act_fn:str="none"

    #---->model paras for training
    is_fixed_adj_matrix:bool = True
    lambda_nll = 0.5
    lambda_reg = 0.5
