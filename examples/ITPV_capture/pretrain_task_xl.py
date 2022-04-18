import argparse
import json
import logging
import os
import random
from io import open
import math
import sys
sys.path.append("../../")
from model.model_GAE import AdaGAE
from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from dataloaders.pretrain_product1m_pv import Pretrain_DataSet_Train, Pretrain_DataSet_Val
from model.capture_IT import BertForMultiModalPreTraining, BertConfig

import torch.distributed as dist

import pdb
from utils_args import get_args
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def dataloader_product1m_train(tokenizer):
    train_dataset = Pretrain_DataSet_Train(
        tokenizer=tokenizer,
        seq_len=36,
        encoding="utf-8",
        predict_feature=False,
        hard_negative=False,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        cache=50000,
        drop_last=False,
        cuda=False,
        distributed=False,
        visualization=False,
        lmdb_file='./dataset/product1m_train_feature.lmdb',
        caption_path='./dataset/caption.json',
        entity_path='./dataset/product1m_entity_data.json',
        unique_entity_dict_path='./dataset/uni_entity.txt',
        MLM=True,
        MRM=True,
        ITM=True
    )
    return train_dataset 

def dataloader_product1m_val(tokenizer):
    test_dataset = Pretrain_DataSet_Val(
        tokenizer=tokenizer,
        seq_len=36,
        encoding="utf-8",
        predict_feature=False,
        hard_negative=False,
        batch_size=8,
        shuffle=False,
        num_workers=16,
        cache=50000,
        drop_last=False,
        cuda=False,
        distributed=False,
        visualization=False,
        lmdb_file='./dataset/train_feature_1000.lmdb',
        caption_path='./dataset/caption.json',
        entity_path='./dataset/product1m_entity_data.json',
        unique_entity_dict_path='./dataset/unique_entity_dict_1000.txt',
    )
    return test_dataset 


def main():

    args = get_args()


    print(args)
    if args.save_name is not '':
        timeStamp = args.save_name
    else:
        timeStamp = strftime("%d-%b-%y-%X-%a", gmtime())
        timeStamp += "_{:0>6d}".format(random.randint(0, 10e6))

    savePath = os.path.join(args.output_dir, timeStamp)

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    config = BertConfig.from_json_file(args.config_file)
    
    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]

    if args.without_coattention:
        config.with_coattention = False
    # save all the hidden parameters. 
    with open(os.path.join(savePath, 'command.txt'), 'w') as f:
        print(args, file=f)  # Python 3.x
        print('\n', file=f)
        print(config, file=f)

    bert_weight_name = json.load(open("../../config/bert-base-uncased_weight_name.json", "r"))
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
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

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    viz = TBlogger("logs", timeStamp)

    print("MLM: {} MRM:{} ITM:{}, CLR:{}".format(args.MLM, args.MRM, args.ITM,args.CLR))


    train_dataset = Pretrain_DataSet_Train(
        tokenizer=tokenizer,
        seq_len=36,
        encoding="utf-8",
        predict_feature=False,
        hard_negative=False,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        cache=50000,
        drop_last=False,
        cuda=False,
        distributed=False,
        visualization=False,
        lmdb_file= args.lmdb_file, #'./dataset/product1m_train_feature.lmdb',
        caption_path= args.caption_path, #'./dataset/caption.json',
        entity_path=args.entity_path, #'./dataset/product1m_entity_data.json',
        unique_entity_dict_path= args.entity_dict, #'./dataset/uni_entity.txt',
        MLM=True,
        MRM=True,
        ITM=True
    )
    print("ds.size(): ",train_dataset.ds.size())

    num_train_optimization_steps = (
        int(
            train_dataset.num_dataset
            / args.train_batch_size
            / args.gradient_accumulation_steps
        )
        * (args.num_train_epochs - args.start_epoch)
    )


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset.num_dataset)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)



    print("Prepare to training!")
    print("MLM: {} MRM:{} ITM:{}".format(args.MLM, args.MRM, args.ITM))

    for epochId in range(int(args.start_epoch), int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataset):
            print("step:{}".format(step))
            continue

        return


class TBlogger:
    def __init__(self, log_dir, exp_name):
        log_dir = log_dir + "/" + exp_name
        print("logging file at: " + log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

    def linePlot(self, step, val, split, key, xlabel="None"):
        self.logger.add_scalar(split + "/" + key, val, step)

if __name__ == "__main__":
    # import time
    # time.sleep(3600*3)

    main()
