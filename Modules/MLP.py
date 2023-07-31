import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

# from GAT implementation
class MLP_ELU_AlphaDrop(nn.Module):
    def __init__(self,
                    #-----> paras for mlp
                    lin_input_dim,fc_dim:list = [64, 48, 32],
                    fc_dropout:float=0.2,
                    #-----> paras for classificer
                    omic_dim:int=32,label_dim:int=2,

                    act_fn:str="none",
                    ):
        super(MLP_ELU_AlphaDrop,self).__init__()


        self.act_fn = define_act_layer(act_type=act_fn)


        fc1 = nn.Sequential(
            nn.Linear(lin_input_dim, fc_dim[0]),
            nn.ELU(),
            nn.AlphaDropout(p=fc_dropout, inplace=True))

        fc2 = nn.Sequential(
            nn.Linear(fc_dim[0], fc_dim[1]),
            nn.ELU(),
            nn.AlphaDropout(p=fc_dropout, inplace=False))

        fc3 = nn.Sequential(
            nn.Linear(fc_dim[1], fc_dim[2]),
            nn.ELU(),
            nn.AlphaDropout(p=fc_dropout, inplace=False))

        fc4 = nn.Sequential(
            nn.Linear(fc_dim[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=fc_dropout, inplace=False))

        self.encoder = nn.Sequential(fc1, fc2, fc3, fc4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))
        
        #-----> for sigmoid activation function only
        self.output_range = nn.Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def run_act_fn(self,x):
        if self.act_fn is not None:
            out = self.act_fn(x)

            if isinstance(self.act_fn, nn.Sigmoid):
                out = x * self.output_range + self.output_shift
        return out

    def forward(self, x):

        #----> graph feature
        GAT_features = x

        features = self.encoder(x)
        out = self.classifier(features)
        #----> embedded feature
        fc_features = features

        if self.act_fn is not None: out = self.act_fn(out)

        return GAT_features, fc_features, out


def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer