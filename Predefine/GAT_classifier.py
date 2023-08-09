from transGNN.Trainer.paras import Trainer_para
from transGNN.Modules.Loss import MixedCoxLossWithReg,CELossWithReg
from transGNN.Trainer.metrics import metrics_para_dict


gat_model_trainer_para= Trainer_para()
gat_model_trainer_para.task_type = "classification"

gat_model_trainer_para.loss_name = "CrossEntropyLoss_with_reg" # "CrossEntropyLoss_with_reg" # "MixedCox_CE_with_reg"
gat_model_trainer_para.loss = CELossWithReg # not create instance MixedCoxLossWithReg
gat_model_trainer_para.loss_para = {
                                    "lambda_nll":1,
                                    "lambda_reg":3e-4,
                                    #"lambda_cox":0.5,
                                    }

gat_model_trainer_para.metrics_names = ["auroc","accuracy","precision"]
gat_model_trainer_para.metrics_paras = {"classification":metrics_para_dict["classification"],}