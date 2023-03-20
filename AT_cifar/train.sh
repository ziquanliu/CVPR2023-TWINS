#!/bin/sh
# export CUDA_VISIBLE_DEVICES=0
lr=0.003
wd=0.00001
python train-AT.py --data_root './data' --data_name 'cifar10' --model_root './adv_pretrain_R50_lr_'$lr'_wd_'$wd'_epoch_60_v1' -w $wd -e 0.0314 --learning_rate $lr -p 'linf' --adv_train --affix 'linf' --log_root './adv_pretrain_R50_lr_'$lr'_wd_'$wd'_epoch_60_v1_log' --gpu '0' -m_e 60 --model-path './resnet50_linf_eps4.0.ckpt' --num_classes 10



python train-TRADES.py --data_root './data' --data_name 'cifar10' --model_root './adv_pretrain_TRADES_R50_lr_'$lr'_wd_'$wd'_epoch_60_v1' -w $wd -e 0.0314 --learning_rate $lr -p 'linf' --adv_train --affix 'linf' --log_root './adv_pretrain_TRADES_R50_lr_'$lr'_wd_'$wd'_epoch_60_v1_log' --gpu '0' -m_e 60 --model-path './resnet50_linf_eps4.0.ckpt' --num_classes 10
