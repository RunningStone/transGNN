import torch
import torch.nn as nn
import torch.nn.functional as F
#------>internal modules
from transGNN.Modules.GAT import GATConvs
from transGNN.Modules.MLP import MLP_ELU_AlphaDrop
from transGNN.Model.GAT.paras import GATParas



class GAT(torch.nn.Module):
    def __init__(self, paras:GATParas):
        super(GAT, self).__init__()
        self.paras = paras
        self.gnn = GATConvs(input_dim=paras.input_dim,
                            nhids=paras.nhids,
                            nheads=paras.nheads,
                            GAT_dropout=paras.GAT_dropout,
                            norm_nb_nodes=paras.norm_nb_nodes,
                            with_layer_norm=paras.with_layer_norm,
                            which_layer=paras.which_layer)
        self.mlp = MLP_ELU_AlphaDrop(lin_input_dim=paras.lin_input_dim,
                                     fc_dim=paras.fc_dim,
                                     fc_dropout=paras.fc_dropout,
                                     omic_dim=paras.omic_dim,
                                     label_dim=paras.class_nb,
                                     act_fn=paras.act_fn)

    def forward(self,data):
        x=self.gnn(data.x,data.edge_index)
        x = torch.flatten(x)
        GAT_features, fc_features, pred = self.mlp(x)
        #print(pred.shape,pred)
        pred = pred.unsqueeze(0)
        Y_hat = torch.argmax(pred, dim=1) 
        Y_prob = F.softmax(pred, dim = 1) 
        results_dict = {'logits': pred,
                        'Y_prob': Y_prob, 
                        'Y_hat': Y_hat,
                        "GAT_features":GAT_features,
                        "fc_features":fc_features,
                        }
        
        return results_dict
    