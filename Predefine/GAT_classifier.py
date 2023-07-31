from transGNN.Trainer.paras import Trainer_para
from transGNN.Modules.Loss import Mixed_CoxLoss_with_reg,CELoss_with_reg,reg_loss,CoxLoss



gat_model_trainer_para= Trainer_para()

gat_model_trainer_para.loss = Mixed_CoxLoss_with_reg