from transGNN.Trainer.paras import Trainer_para
from transGNN.Modules.Loss import MixedCoxLossWithReg,CELossWithReg



gat_model_trainer_para= Trainer_para()

gat_model_trainer_para.loss_name = "MixedCox_CE_with_reg" # "CrossEntropyLoss_with_reg" # "MixedCox_CE_with_reg"
gat_model_trainer_para.loss = MixedCoxLossWithReg # not create instance
gat_model_trainer_para.loss_para = {
                                    "lambda_nll":0.5,
                                    "lambda_reg":0.5,
                                    "lambda_cox":0.5,
                                    }