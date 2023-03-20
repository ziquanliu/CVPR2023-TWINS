#!/bin/sh
# export CUDA_VISIBLE_DEVICES=0
lr=0.003
wd=0.00001
lambda_twins=1.0
python train-TWINS-AT.py --data_root './data' --data_name 'cifar10' --model_root './adv_pretrain_two_diff_batch_R50_lr_'$lr'_wd_'$wd'_lambda_one_lambda_twins_'$lambda_twins'_epoch_60_v1' -w $wd -e 0.0314 --learning_rate $lr -p 'linf' --adv_train --affix 'linf' --log_root './adv_pretrain_two_diff_batch_R50_lr_'$lr'_wd_'$wd'_lambda_one_lambda_twins_'$lambda_twins'_epoch_60_v1_log' --gpu '0' -m_e 60 --model-path './resnet50_linf_eps4.0.ckpt' --num_classes 10 --lambda_twins $lambda_twins



python train-TWINS-TRADES.py --data_root './data' --data_name 'cifar10' --model_root './adv_pretrain_two_diff_batch_TRADES_R50_lr_'$lr'_wd_'$wd'_lambda_one_lambda_twins_'$lambda_twins'_epoch_60_v1' -w $wd -e 0.0314 --learning_rate $lr -p 'linf' --adv_train --affix 'linf' --log_root './adv_pretrain_two_diff_batch_TRADES_R50_lr_'$lr'_wd_'$wd'_lambda_one_lambda_twins_'$lambda_twins'_epoch_60_v1_log' --gpu '0' -m_e 60 --model-path './resnet50_linf_eps4.0.ckpt' --num_classes 10 --lambda_twins $lambda_twins
