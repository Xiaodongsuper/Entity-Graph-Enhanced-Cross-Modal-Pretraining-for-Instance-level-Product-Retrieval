#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pretrain_task.py \
 --from_pretrained ../../bert_model/bert_base_chinese \
 --bert_model bert-base-chinese \
 --config_file ../../config/capture.json\
 --predict_feature \
 --learning_rate 1e-4 \
 --train_batch_size 64 \
 --max_seq_length 36 \
 --lmdb_file ../../dataset_tongkuan/train_feature.lmdb \
 --caption_path ../../dataset_tongkuan/train_caption.json \
 --entity_path ../../dataset_tongkuan/product1m_entity_data.json \
 --entity_dict ../../dataset_tongkuan/unique_entity_dict_all_19w.txt \
 --num_train_epochs 5 \
 --save_name 15w_text_you_focus_fixed_test_training\
 --MLM \
 --MRM \
 --CLR