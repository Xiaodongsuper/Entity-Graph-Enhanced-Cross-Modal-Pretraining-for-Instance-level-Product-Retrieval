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
import pickle
from dataloaders.pretrain_product1m_pv import Pretrain_DataSet_Train, Pretrain_DataSet_Val
from model.capture_IT_graph_text_you import BertForMultiModalPreTraining, BertConfig #capture_IT_graph_text_you
import torch.nn.functional as F
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

def write_pickle(filename,data):
    open(filename,"wb").write(pickle.dumps(data))
    return

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
    

    num_train_optimization_steps = (
        int(
            train_dataset.num_dataset
            / args.train_batch_size
            / args.gradient_accumulation_steps
        )
        * (args.num_train_epochs - args.start_epoch)
    )

    default_gpu = False
    if dist.is_available() and args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    # pdb.set_trace()
    if args.predict_feature:
        config.v_target_size = 1 #2048 feature dim
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False

    if args.from_pretrained:
        model = BertForMultiModalPreTraining.from_pretrained(args.from_pretrained, config)
    else:
        model = BertForMultiModalPreTraining(config)


    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset.num_dataset)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    startIterID = 0
    global_step = 0
    masked_loss_v_tmp = 0
    masked_loss_t_tmp = 0
    entity_contras_loss_tmp = 0
    entity_graph_contras_loss_tmp = 0
    next_sentence_loss_tmp = 0
    loss_tmp = 0
    start_t = timer()


    print("Prepare to training!")
    print("MLM: {} MRM:{} ITM:{}".format(args.MLM, args.MRM, args.ITM))
    right_temp=0
    all_num_temp=0
    neighbors = 3
    lam = 2e-2
    f = open(args.entity_dict, 'r')
    entity_node_fea = np.ones([len(f.readlines()), 768])*1/768
    update_entity_graph_fea = None
    all_t_fea = []
    all_v_fea = []
    
    lib_t_feature_np=np.zeros((1,1024))
    lib_v_feature_np=np.zeros((1,1024))
    lib_vil_feature_np=np.zeros((1,1024))
    lib_vil_id=[]

    t_feature_list=[]
    v_feature_list=[]
    vil_feature_list=[]
    model.load_state_dict(torch.load('./save/15w_configure_066/pytorch_model_4.bin')) #15w_no_graph_text_you
    model.cuda()
    model.eval()
    with torch.no_grad():
        i = 0
        for step, batch in enumerate(tqdm(train_dataset)):
            image_id=batch[-1]
            batch=batch[:-1]
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            input_ids, input_mask, segment_ids, pairs_masked_text, pairs_token_labels, is_next,\
            entity_ids,  entity_mask, entity_segment_ids, pairs_entity_masked_text, pairs_entity_token_labels, \
            each_entity_text, each_entity_mask, each_entity_segment, entity_index, \
            image_feat, masked_video, image_loc, image_label, image_target, image_mask =batch
            _,_, \
            _, \
            pooled_output_t, \
            pooled_output_v = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
                pairs_token_labels,
                image_label,
                image_target,
                is_next,
                entity_ids,
                entity_mask,
                entity_segment_ids,
                pairs_entity_token_labels,
                each_entity_text,
                each_entity_mask,
                each_entity_segment,
                entity_index,
                entity_sim=None,
                output_all_attention_masks=False,
                return_features=True
            )
            lib_vil_id+=list(image_id)
            #pooled_output_vil= pooled_output_v*pooled_output_t
            
            #normalize 
            pooled_output_t = F.normalize(pooled_output_t, dim=-1)
            pooled_output_v = F.normalize(pooled_output_v, dim=-1)
            
            pooled_output_t=pooled_output_t.detach().cpu().numpy()
            pooled_output_v=pooled_output_v.detach().cpu().numpy()
            pooled_output_vil = np.hstack((pooled_output_v,pooled_output_t)) #=pooled_output_vil.detach().cpu().numpy()

            t_feature_list.append(pooled_output_t)
            v_feature_list.append(pooled_output_v)
            vil_feature_list.append(pooled_output_vil)
        #pdb.set_trace()
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
    #pdb.set_trace()
    write_pickle("{}/t_feature_np.p".format(args.feature_dir),lib_t_feature_np)
    write_pickle("{}/v_feature_np.p".format(args.feature_dir),lib_v_feature_np)
    write_pickle("{}/vil_feature_np.p".format(args.feature_dir),lib_vil_feature_np)
    write_pickle("{}/id.p".format(args.feature_dir),lib_vil_id)
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
