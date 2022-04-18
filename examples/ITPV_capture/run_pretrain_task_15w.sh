#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pretrain_task.py \
 --from_pretrained ../../bert_model/bert_base_chinese \
 --bert_model bert-base-chinese \
 --config_file ../../config/capture_change.json\
 --predict_feature \
 --learning_rate 1e-4 \
 --train_batch_size 64 \
 --max_seq_length 36 \
 --lmdb_file ../../dataset/product1m_train15w_feature.lmdb \
 --caption_path ../../dataset/caption.json \
 --entity_path ../../dataset/product1m_entity_data.json \
 --entity_dict ../../dataset/unique_entity_dict_all_15w.txt \
 --num_train_epochs 5 \
 --save_name 15w_text_you_focus_fixed_test_normlize_training\
 --MLM \
 --MRM \
 --CLR