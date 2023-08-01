import networkx as nx
from torch_geometric.utils import from_networkx




def get_pyg_graph_data(adj_M,feature,label):
    """
    pyg graph data from adj matrix, feature of each node, label of each node
    in:
        adj_M:np.ndarray: adj matrix
        feature:torch.tensor   
        label:torch.tensor
    """
    device = label.device
    # create graph for networkx
    G = nx.from_numpy_matrix(adj_M,create_using=nx.MultiGraph)
    # create pyg graph data for network
    data = from_networkx(G)
    #print(feature.shape,label.shape)
    data.x = feature.transpose(0,1)
    data.y = label
    data = data.to(device)
    return data