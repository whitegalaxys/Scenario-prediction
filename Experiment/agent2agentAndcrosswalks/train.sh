python train.py \
--name DIPP \
--train_set ../dataset/processed_train \
--valid_set ../dataset/processed_val \
--seed 3407 \
--num_workers 8 \
--pretrain_epochs 4 \
--train_epochs 8 \
--batch_size 12 \
--learning_rate 2e-4 \
--device cuda

