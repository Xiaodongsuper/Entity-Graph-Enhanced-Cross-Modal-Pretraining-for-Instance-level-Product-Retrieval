import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

import sys
sys.path.append("../../../")

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle

from pytorch_pretrained_bert.tokenization import BertTokenizer
from dataloaders.pretrain_product1m_pv import Pretrain_DataSet_Train, Pretrain_DataSet_Val
from model.capture_IT import BertForMultiModalPreTraining, BertConfig
from utils_args import get_args

import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_pickle(filename):
    return pickle.loads(open(filename,"rb").read())

def write_pickle(filename,data):
    open(filename,"wb").write(pickle.dumps(data))
    return


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # timeStamp = '-'.join(task_names) + '_' + args.config_file.split('/')[1].split('.')[0]
    if '/' in args.from_pretrained:
        timeStamp = args.from_pretrained.split('/')[1]
    else:
        timeStamp = args.from_pretrained

    savePath = os.path.join(args.output_dir, timeStamp)

    config = BertConfig.from_json_file(args.config_file)
    # bert_weight_name = json.load(open("config/" + args.bert_model + "_weight_name.json", "r"))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    # if default_gpu and not os.path.exists(savePath):
    #     os.makedirs(savePath)

    # task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val \
    #     = LoadDatasetEval(args, task_cfg, args.tasks.split('-'))
    #

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    train_dataset = ConceptCapLoaderVal_task(
        args.train_file,
        tokenizer,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        predict_feature=args.predict_feature,
        num_workers=args.num_workers,
        distributed=args.distributed,
        lmdb_file=args.lmdb_file, # '/data3/xl/vilbert/data/ali2/SP_train.lmdb'
        caption_path=args.caption_path,  #"/data3/xl/MP/data2/id_info_dict.json"
        MLM = args.MLM,
        MRM = args.MRM,
        ITM = args.ITM
    )

    print("all image batch num: ", len(train_dataset))

    config.fast_mode = True
    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False

    if args.without_coattention:
        config.with_coattention = False

    model = BertForMultiModalPreTraining_SimCLR.from_pretrained(args.from_pretrained, config)

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, deay_allreduce=True)

    elif n_gpu > 1:
        model = nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    print("Prepare to generate feature! ready!")
    model.eval()



    # lib
    lib_t_feature_np=np.zeros((1,1024))
    lib_v_feature_np=np.zeros((1,1024))
    lib_vil_feature_np=np.zeros((1,1024))
    lib_vil_id=[]

    t_feature_list=[]
    v_feature_list=[]
    vil_feature_list=[]

    for step, batch in enumerate(tqdm(train_dataset)):

        image_ids = batch[-1]
        batch = batch[:-1]
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask = (
            batch
        )

        if torch.sum(torch.isnan(image_feat)) > 0:
            continue

        # lib_vil_id.append(image_ids)
        lib_vil_id+=list(image_ids)


        for image_id in image_ids:
            if image_id=="626253996983":
                print("here")

        with torch.no_grad():
            masked_loss_t, masked_loss_v, next_sentence_loss,pooled_output_t,pooled_output_v = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
                lm_label_ids,
                image_label,
                image_target,
                is_next,
                return_features=True,
                return_hidden=args.return_hidden
            )

        pooled_output_vil=pooled_output_v*pooled_output_t

        pooled_output_t=pooled_output_t.detach().cpu().numpy()
        pooled_output_v=pooled_output_v.detach().cpu().numpy()
        pooled_output_vil=pooled_output_vil.detach().cpu().numpy()

        t_feature_list.append(pooled_output_t)
        v_feature_list.append(pooled_output_v)
        vil_feature_list.append(pooled_output_vil)


        # if step==0:
        #     lib_t_feature_np=pooled_output_t
        #     lib_v_feature_np=pooled_output_v
        #     lib_vil_feature_np=pooled_output_vil
        # else:
        #     lib_t_feature_np=np.vstack((lib_t_feature_np,pooled_output_t))
        #     lib_v_feature_np=np.vstack((lib_v_feature_np,pooled_output_v))
        #     lib_vil_feature_np=np.vstack((lib_vil_feature_np,pooled_output_vil))


        # if step==10000:
        #     break

    lib_t_feature_np=np.vstack(t_feature_list)
    lib_v_feature_np=np.vstack(v_feature_list)
    lib_vil_feature_np=np.vstack(vil_feature_list)

    print("lib_t_feature_np: ",lib_t_feature_np.shape)
    print("lib_v_feature_np: ",lib_v_feature_np.shape)
    print("lib_vil_feature_np: ",lib_vil_feature_np.shape)

    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)

    write_pickle("{}/t_feature_np.p".format(args.feature_dir),lib_t_feature_np)
    write_pickle("{}/v_feature_np.p".format(args.feature_dir),lib_v_feature_np)
    write_pickle("{}/vil_feature_np.p".format(args.feature_dir),lib_vil_feature_np)
    write_pickle("{}/id.p".format(args.feature_dir),lib_vil_id)




if __name__ == "__main__":
    main()
