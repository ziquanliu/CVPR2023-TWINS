#!/bin/sh
# export CUDA_VISIBLE_DEVICES=0
lr=0.003
wd=0.001
lambda_twins=0.4
python train-TWINS-AT.py --data_root './caltech-256' --model_root './adv_pretrain_R50_two_diff_batch_lr_'$lr'_wd_'$wd'_lr_decay_lambda_one_lambda_twins_'$lambda_twins'_epoch_60_v2' -w $wd -e 0.0314 --learning_rate $lr -p 'linf' --adv_train --affix 'linf' --log_root './adv_pretrain_R50_two_diff_batch_lr_'$lr'_wd_'$wd'_lr_decay_lambda_one_lambda_twins_'$lambda_twins'_epoch_60_v2_log' --gpu '0' -m_e 60 --model-path './resnet50_linf_eps4.0.ckpt' --num_classes 257 --lambda_twins $lambda_twins

python train-TWINS-TRADES.py --data_root './caltech-256' --model_root './adv_pretrain_R50_two_diff_batch_TRADES_lr_'$lr'_wd_'$wd'_lr_decay_lambda_one_lambda_twins_'$lambda_twins'_epoch_60_v2' -w $wd -e 0.0314 --learning_rate $lr -p 'linf' --adv_train --affix 'linf' --log_root './adv_pretrain_R50_two_diff_batch_TRADES_lr_'$lr'_wd_'$wd'_lr_decay_lambda_one_lambda_twins_'$lambda_twins'_epoch_60_v2_log' --gpu '0' -m_e 60 --model-path './resnet50_linf_eps4.0.ckpt' --num_classes 257 --lambda_twins $lambda_twins
