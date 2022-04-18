#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5,6 python pretrain_task_no_graph.py \
 --from_pretrained ../../bert_model/bert_base_chinese \
 --bert_model bert-base-chinese \
 --config_file ../../config/capture.json\
 --predict_feature \
 --learning_rate 1e-4 \
 --train_batch_size 64 \
 --max_seq_length 36 \
 --lmdb_file ../../dataset/product1m_train15w_feature.lmdb \
 --caption_path ../../dataset/caption.json \
 --entity_path ../../dataset/product1m_entity_data.json \
 --entity_dict ../../dataset/unique_entity_dict_all_15w.txt \
 --num_train_epochs 5 \
 --save_name capture_subset_v2_MLM_MRM_CLR \
 --output_dir ./save/15w_no_graph_attention_text_attention_you \
 --MLM \
 --MRM \
 --CLR