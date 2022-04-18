#!/usr/bin/env bash
TRAIN_TYPE=IT_capture
MODEL_TYPE=capture_retrievalset_MLM_MRM_CLR
QUERY_FEATURE=query_feature_normal
GALLERY_FEATURE=gallery_feature_normal
QUERY_FEATURE_DIR=examples/${TRAIN_TYPE}/eval/feature_data/${MODEL_TYPE}/${QUERY_FEATURE}
GALLERY_FEATURE_DIR=examples/${TRAIN_TYPE}/eval/feature_data/${MODEL_TYPE}/${GALLERY_FEATURE}
RETRIEVAL_RESULTS_DIR=examples/${TRAIN_TYPE}/eval/retrieval_id_list/${MODEL_TYPE}/${QUERY_FEATURE}



# remenber to change the dir and filename
# # gallery
CUDA_VISIBLE_DEVICES=7 python extract_feature.py \
  --from_pretrained ../../bert_model/bert_base_chinese \
  --bert_model bert-base-chinese \
  --config_file ../../config/capture.json \
  --predict_feature \
  --learning_rate 1e-4 \
  --train_batch_size 64 \
  --max_seq_length 36 \
  --lmdb_file ../../dataset_tongkuan/query_feature.lmdb \
  --caption_path ../../dataset_tongkuan/query_caption.json \
  --entity_path ../../dataset_tongkuan/test_entity_data.json \
  --entity_dict ../../dataset_tongkuan/unique_entity_dict_all_19w.txt \
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
  --lmdb_file ../../dataset_tongkuan/gallery_feature.lmdb \
  --caption_path ../../dataset_tongkuan/gallery_caption.json \
  --entity_path ../../dataset_tongkuan/gallery_entity_data.json \
  --entity_dict ../../dataset_tongkuan/unique_entity_dict_all_19w.txt  \
  --num_train_epochs 5 \
  --save_name capture_subset_v2_MLM_MRM_CLR \
  --feature_dir ./feature/gallery




# python retrieval_unit_id_list_v2.py \
#  --query_feature_path ./feature/test \
#  --gallery_feature_path ./feature/gallery \
#  --retrieval_results_path ./feature/results \
#  --v \
#  --t \
#  --vil




# OUTPUT_METRIC_DIR=./metric_results/
# python evaluate_suit.py \
#   --retrieval_result_dir ./feature/results \
#   --GT_dir /data2/xiaodong/Product1m/all_data/ \
#   --output_metric_dir ${OUTPUT_METRIC_DIR} \
#   --v \
#   --t \
#   --vil


# cd ../../..


python retrieval_unit_id_list_v2.py \
  --query_feature_path ./feature/test \
  --gallery_feature_path ./feature/gallery \
  --retrieval_results_path ./feature/results \
  --v \
  --t \
  --vil \

  # --vax_topk 10 \


GT_file=/data1/xl/multi_modal/Retrieval/data/raw_data/v2/retrieval_id_info.json
OUTPUT_METRIC_DIR=./metric_results/vil/
python evaluate_unit_v2.py \
  --retrieval_result_dir ./feature/results \
  --GT_file ${GT_file} \
  --output_metric_dir ${OUTPUT_METRIC_DIR} \
  --t \
  --v \
  --vil \


#
#RETRIEVAL_IMAGES_DIR=examples/${TRAIN_TYPE}/eval/retrieval_images/${MODEL_TYPE}/${QUERY_FEATURE}
#python retrieval_unit_images.py \
#  --retrieval_ids_path ${RETRIEVAL_RESULTS_DIR} \
#  --retrieval_images_path ${RETRIEVAL_IMAGES_DIR} \
#  --query_image_prefix  /data1/xl/multi_modal/Retrieval/data/images \
#  --gallery_image_prefix /data1/xl/multi_modal/Retrieval/data/images \
#  --i
#












