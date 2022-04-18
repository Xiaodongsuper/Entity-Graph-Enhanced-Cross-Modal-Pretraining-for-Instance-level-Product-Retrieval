#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pretrain_task.py \
 --from_pretrained ../../bert_model/bert_base_chinese \
 --bert_model bert-base-chinese \
 --config_file ../../config/capture.json\
 --predict_feature \
 --learning_rate 1e-4 \
 --train_batch_size 128 \
 --max_seq_length 36 \
 --lmdb_file ../../dataset/product1m_train_feature.lmdb \
 --caption_path ../../dataset/caption.json \
 --entity_path ../../dataset/product1m_entity_data.json \
 --entity_dict ../../dataset/uni_entity.txt \
 --num_train_epochs 5 \
 --save_name capture_subset_v2_MLM_MRM_CLR \
 --start_epoch 0 \
 --MLM \
 --MRM \
 --CLR