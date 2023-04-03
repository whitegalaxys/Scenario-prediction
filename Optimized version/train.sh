python train.py \
--name DIPP \
--train_set ../dataset/processed_train \
--valid_set ../dataset/processed_val \
--seed 3407 \
--num_workers 8 \
--pretrain_epochs 5 \
--train_epochs 20 \
--batch_size 13 \
--learning_rate 2e-4 \
--device cuda

