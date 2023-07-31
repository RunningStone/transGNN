"""
Model from Multi-Level Attention Graph Neural Network Based on Co-expression Gene Modules for Disease Diagnosis and Prognosis
github: https://github.com/TencentAILabHealthcare/MLA-GNN

"""
import pandas as pd
import torch

#import torchsnooper
from histocore.DATA.Database.utils import get_pyg_graph_data
#---->
from transGNN import logger
from transGNN.Trainer.pl_base import pl_basic
from transGNN.Trainer.paras import Trainer_para
from transGNN.Model.GAT.paras import GATParas

#---->
###################################################################
#           
###################################################################
class pl_GAT(pl_basic):
    def __init__(self,
                trainer_paras:Trainer_para, # trainer para
                model_paras:GATParas) -> None:
        super().__init__(self,trainer_paras=trainer_paras,model_paras=model_paras)

        self.M  = None
        self.lambda_nll = self.model_paras.lambda_nll
        self.lambda_reg = self.model_paras.lambda_reg

        self.loss_name = self.opt_paras.Loss_name[0]

    def set_adj_M(self,M):
        # set adjacent matrix
        self.M = M#torch.from_numpy(M)
        #self.M.requires_grad = False
        
    def get_graph_data(self,M,feature,label):
        """
        graph include three parts: adj_matrix, feature of each node, label for graph
        """
        data = get_pyg_graph_data(M,feature,label)
        return data   

    def special_loss(self,pred,label):
        if self.loss_name == "CrossEntropyLoss_with_reg" \
                  or self.loss_name == "MixedCox_CE_with_reg":
            L=self.loss(self.model,
                        pred,
                        label,
                        self.lambda_nll,
                        self.lambda_reg)
        else:
            raise ValueError(f" target loss function {self.loss_name} not fit the model.")
        return L 

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward

        x, y = batch
        y = y.long()
        # model for CE loss
        if self.model_paras.is_fixed_adj_matrix:
            pyg_data = self.get_graph_data(self.M,x,y)
        else:
            # TODO: add the dynamic adj matrix
            logger.warning("dynamic adj matrix is not implemented yet.")
            pyg_data = self.get_graph_data(self.M,x,y)
        results_dict = self.model(pyg_data)

        #---->loss
        loss = self.special_loss(results_dict['logits'],y)

        #---->overall counts log 
        self.counts_step(Y_hat=results_dict['Y_hat'], label=y, train_phase="train")
        return {'loss': loss} 


    def validation_step(self, batch, batch_idx):
        data, label = batch
        label = label.long()

        # model for CE loss
        if self.model_paras.is_fixed_adj_matrix:
            pyg_data = self.get_graph_data(self.M,data,label)
        else:
            # TODO: add the dynamic adj matrix
            logger.warning("dynamic adj matrix is not implemented yet.")
            pyg_data = self.get_graph_data(self.M,data,label)
        results_dict = self.model(pyg_data)

        results_dict.update({'label' : label})
        return results_dict


    
