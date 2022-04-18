#!/usr/bin/env bash

#
CUDA_VISIBLE_DEVICES=7 python extract_feature.py \
  --from_pretrained ../../bert_model/bert_base_chinese \
  --bert_model bert-base-chinese \
  --config_file ../../config/capture.json \
  --predict_feature \
  --learning_rate 1e-4 \
  --train_batch_size 64 \
  --max_seq_length 36 \
  --lmdb_file ../../dataset/test_feature.lmdb \
  --caption_path ../../dataset/caption.json \
  --entity_path ../../dataset/test_entity_data.json \
  --entity_dict ../../dataset/unique_entity_dict_1000.txt \
  --num_train_epochs 5 \
  --save_name capture_subset_v2_MLM_MRM_CLR \
  --feature_dir ./feature/test


CUDA_VISIBLE_DEVICES=7 python extract_feature.py \
  --from_pretrained ../../bert_model/bert_base_chinese \
  --bert_model bert-base-chinese \
  --config_file ../../config/capture.json \
  --predict_feature \
  --learning_rate 1e-4 \
  --train_batch_size 64 \
  --max_seq_length 36 \
  --lmdb_file ../../dataset/gallery_feature.lmdb \
  --caption_path ../../dataset/caption.json \
  --entity_path ../../dataset/gallery_entity_data.json \
  --entity_dict ../../dataset/unique_entity_dict_1000.txt \
  --num_train_epochs 5 \
  --save_name capture_subset_v2_MLM_MRM_CLR \
  --feature_dir ./feature/gallery




python retrieval_unit_id_list.py \
 --query_feature_path ./feature/test \
 --gallery_feature_path ./feature/gallery \
 --retrieval_results_path ./feature/results \
 --v \
 --t \
 --vil




OUTPUT_METRIC_DIR=./metric_results/
python evaluate_suit.py \
  --retrieval_result_dir ./feature/results \
  --GT_dir /data2/xiaodong/Product1m/all_data/ \
  --output_metric_dir ${OUTPUT_METRIC_DIR} \
  --v \
  --t \
  --vil






# OUTPUT_METRIC_DIR=./metric_results/
# python evaluate_unit.py \
#  --retrieval_result_dir ./feature/results \
#  --GT_dir /data2/xiaodong/Product1m/all_data/ \
#  --output_metric_dir ${OUTPUT_METRIC_DIR} \
#  --v \
#  --t \
#  --vil




