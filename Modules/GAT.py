import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class GATConvs(nn.Module):
    def __init__(self,
                #------> GAT Conv layers
                input_dim,nhids:list=[8, 16, 12],nheads:list=[4, 3, 4],GAT_dropout:float=0.2,                    
                #---> paras for layer norm
                norm_nb_nodes:int=10,with_layer_norm:bool=True,which_layer:str="all",
                ) -> None:
        super(GATConvs,self).__init__()

        self.conv1 = pyg_nn.GATConv(input_dim, nhids[0], heads=nheads[0],dropout=GAT_dropout)

        self.conv2 = pyg_nn.GATConv(nhids[0]*nheads[0], nhids[1], heads=nheads[1],dropout=GAT_dropout)

        self.conv3 = pyg_nn.GATConv(nhids[1]*nheads[1], nhids[2], heads=nheads[2],dropout=GAT_dropout)

        self.pool1 = torch.nn.Linear(nhids[0]*nheads[0], 1)
        self.pool2 = torch.nn.Linear(nhids[1]*nheads[1], 1)
        self.pool3 = torch.nn.Linear(nhids[2]*nheads[2], 1)

        self.with_layer_norm = with_layer_norm
        self.which_layer = which_layer
        self.layer_norm0 = nn.LayerNorm(norm_nb_nodes)
        self.layer_norm1 = nn.LayerNorm(norm_nb_nodes)
        self.layer_norm2 = nn.LayerNorm(norm_nb_nodes)
        self.layer_norm3 = nn.LayerNorm(norm_nb_nodes)

    def layer_norm(self,x0,x1,x2):
        if self.with_layer_norm == "True":
            x0 = self.layer_norm0(x0)
            x1 = self.layer_norm1(x1)
            x2 = self.layer_norm0(x2)

        if self.which_layer == 'all':
            x = torch.cat([x0, x1, x2], dim=1)
        elif self.which_layer == 'layer1':
            x = x0
        elif self.which_layer == 'layer2':
            x = x1
        elif self.which_layer == 'layer3':
             x = x2
        else:
            raise ValueError(f"{self.which_layer} not in list [all,layer1,layer2,layer3]")
        return x

    def forward(self,x,adj):
        """
        input: mini-batch input. size: [batch_size, num_nodes, node_feature_dim]
        adj:   adjacency matrix. size: [num_nodes, num_nodes].  need to be expanded to batch_adj later.
        """
        batch = torch.linspace(0, x.size(0) - 1, x.size(0), dtype=torch.long)
        batch = batch.unsqueeze(1).repeat(1, x.size(1)).view(-1).to(x.device)

        ### layer1
        x = x.requires_grad_()
        x0 = torch.mean(x, dim=-1)
        x0 = pyg_utils.to_dense_batch(x0, batch=batch)[0] #[bs, nodes]

        ### layer2
        x = F.dropout(x, p=0.2, training=self.training)
        #print(x.shape,adj.shape)
        x = self.conv1(x, adj)
        x = F.elu(x) #[bs*nodes, nhids[0]*nheads[0]]
        x1 = self.pool1(x).squeeze(-1)
        x1 = pyg_utils.to_dense_batch(x1, batch=batch)[0] #[bs, nodes]

        ### layer3
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv2(x, adj))  # [bs*nodes, nhids[0]*nheads[0]]
        x2 = self.pool2(x).squeeze(-1)
        x2 = pyg_utils.to_dense_batch(x2, batch=batch)[0]  # [bs, nodes]

        #print(x0.shape,x1.shape,x2.shape)
        x=self.layer_norm(x0,x1,x2)
        return x

