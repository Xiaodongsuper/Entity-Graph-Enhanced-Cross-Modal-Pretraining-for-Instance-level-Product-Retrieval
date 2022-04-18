import copy
import json
import logging
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td
from tensorpack.dataflow import MultiThreadMapData
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb
import re
from string import punctuation
from torch.utils.data import DataLoader, Dataset, sampler
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

def write_json(file,data):
    f=open(file,"w",encoding="utf-8")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return


# pairs_text, pairs_mask, pairs_segment, bbox_image, image_mask, \
# pairs_masked_text, pairs_token_labels, masked_image, image_labels_index, image_location, \
# pairs_entity_text, pairs_entity_mask, pairs_entity_segment, pairs_entity_masked_text, pairs_entity_token_labels, pairs_each_entity_text, pairs_each_entity_mask, pairs_each_entity_segment, entity_index

class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self, 
        image_id=None,
        image_feat=None,
        image_target=None,
        caption=None,
        masked_caption=None,
        entity_concated_ids=None,
        each_entity_ids=None,
        each_entity=None,
        is_next=None,
        lm_labels=None,
        image_loc=None,
        num_boxes=None
    ):
        self.image_id = image_id
        self.image_feat = image_feat
        self.caption = caption
        self.each_entity = each_entity
        self.masked_caption=masked_caption,
        self.entity_concated_ids=entity_concated_ids,
        self.each_entity_ids=each_entity_ids,
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes





class InputFeatures(object):
    """A single set of features of data."""
    def __init__(
        self,
        image_id=None,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        masked_input_ids=None,
        lm_label_ids = None,
        is_next=None,
        entity_concated_ids=None,
        entity_concated_mask=None,
        entity_concated_segment_ids=None,
        masked_concated_ids=None,
        lm_entity_label_ids=None,
        entity_each_ids=None,
        entity_each_mask=None,
        entity_each_segment=None,
        entity_each_index=None,
        image_feat=None,
        masked_image_feat=None,
        image_target=None,
        image_loc=None,
        image_label=None,
        image_mask=None,
    ):
        self.image_id = image_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_target = image_target
        self.image_mask = image_mask
        self.entity_concated_ids = entity_concated_ids
        self.entity_concated_mask = entity_concated_mask
        self.entity_concated_segment_ids = entity_concated_segment_ids 
        self.masked_concated_ids=masked_concated_ids
        self.lm_entity_label_ids=lm_entity_label_ids
        self.entity_each_mask=entity_each_mask
        self.entity_each_segment=entity_each_segment
        self.entity_each_ids = entity_each_ids
        self.entity_each_index = entity_each_index
        self.masked_input_ids = masked_input_ids
        self.masked_image_feat = masked_image_feat


class BertPreprocessBatch(object):
    def __init__(
        self,
        caption_path,
        entity_path,
        unique_entity_dict_path,
        tokenizer,
        seq_len,
        region_len, 
        data_size,
        split="Train",
        encoding="utf-8",
        predict_feature=False,
        visualization=False,
        MLM=True,
        MRM=True,
        ITM=True
    ):

        self.MLM=MLM
        self.MRM=MRM
        self.ITM=ITM

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.predict_feature = predict_feature

        # self.captions = list(json.load(open(caption_path, 'r')).values())
        id_info_dict = json.load(open(caption_path, 'r'))
        entity_data = read_json(entity_path)
        entity_dict = []
        f = open(unique_entity_dict_path,'r')
        for entity_i in f.readlines():
            entity_dict.append(entity_i.split('\n')[0])
        self.captions=[]  # TODO: change
        for each in id_info_dict:
            self.captions.append(id_info_dict[each]["sentences"])
        self.entity_dict = entity_dict
        self.entity_data = entity_data
            # pv_pairs=id_info_dict[each]["pv_pairs"].split("#;#")
            # if len(pv_pairs)==1:
            #     pv_pairs_str=""
            # else:
            #     pv_pairs_str=" ".join([pv.split("#:#")[1] for pv in pv_pairs])
            # self.captions.append(id_info_dict[each]["title"] + pv_pairs_str)
        # print("hello here! ")
        self.num_caps=len(self.captions)
        self.visualization = visualization

    def __call__(self, data):

        image_feature_wp, image_location_wp, num_boxes,  image_h, image_w, image_id, caption  = data
        try:
            all_entity = self.entity_data[image_id]['entity']
        except:
            all_entity = []
        if(all_entity==[]):
            all_entity = ['']
        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        # image_target[:num_boxes] = image_target_wp
        image_location[:num_boxes,:4] = image_location_wp

        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)

        if self.predict_feature:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)   

        punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——【】{};；●，。&～、|\s:：'
        caption = re.sub(r"[{}]+".format(punc)," ", caption)

        tokens_caption = self.tokenizer.tokenize(caption)
        caption, masked_label = self.random_cap(caption)
        
        entity_list = [''.join(each) for each in all_entity]
        entity_list = '[SEP]'.join(entity_list)
        entity_concated_ids = self.tokenizer.tokenize(entity_list)
        each_entity_ids = []
        each_entity = []
        for entity_i in all_entity:
            each_entity_ids.append(self.tokenizer.tokenize(entity_i))
            each_entity.append(entity_i)
        # print("len caption: ",len(tokens_caption))
        cur_example = InputExample(
            image_id=image_id,
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            entity_concated_ids=entity_concated_ids,
            each_entity_ids=each_entity_ids,
            each_entity=each_entity,
            is_next=masked_label,
            image_loc=image_location,
            num_boxes=num_boxes
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)

        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.masked_input_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.entity_concated_ids,
            cur_features.entity_concated_mask,
            cur_features.entity_concated_segment_ids,
            cur_features.masked_concated_ids,
            cur_features.lm_entity_label_ids,
            cur_features.entity_each_ids,
            cur_features.entity_each_mask,
            cur_features.entity_each_segment,
            cur_features.entity_each_index,
            cur_features.image_feat,
            cur_features.masked_image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
            cur_features.image_id,
        )
        return cur_tensors

    def random_cap(self, caption):
        if self.visualization:
            return caption, 0

        if self.ITM:
            if random.random() > 0.5:
                label = 0
            else:
                caption = self.get_random_caption()
                label = 1
        else:
            label = 0

        return caption, label

    def get_random_caption(self):
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        caption = self.captions[rand_doc_idx]

        return caption

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):
        image_id = example.image_id
        image_feat = example.image_feat
        caption = example.caption
        entity_concated_ids = example.entity_concated_ids[0]
        each_entity_ids = example.each_entity_ids[0]
        each_entity = example.each_entity
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        self._truncate_seq_pair(caption, max_seq_length - 2)

        masked_caption, masked_caption_label = self.random_word(caption, tokenizer)
        masked_concated_entity, masked_concated_entity_label = self.random_word(entity_concated_ids, tokenizer)
        original_feat = image_feat
        masked_image_feat, image_loc, image_label = self.random_region(image_feat, image_loc, num_boxes)

        # concatenate lm labels and account for CLS, SEP, SEP
        # lm_label_ids = ([-1] + caption_label + [-1] + image_label + [-1])
        #lm_label_ids = [-1] + caption_label + [-1]
        lm_label_ids = [-1] + masked_caption_label + [-1]
        lm_entity_label_ids = [-1] + masked_concated_entity_label + [-1]
        # image_label = ([-1] + image_label)

        ## caption masked caption imgae
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = input_ids[0:max_seq_length]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            
            #lm_label_ids.append(-1)

        masked_tokens = []
        masked_tokens.append("[CLS]")
        for token in masked_caption:
            masked_tokens.append(token)
        masked_tokens.append("[SEP]")


        masked_input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
        masked_input_ids = masked_input_ids[0:max_seq_length]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]

        while len(masked_input_ids) < max_seq_length:
            masked_input_ids.append(0)
            lm_label_ids.append(-1)

        
        ### entity concated
        entity_concated_tokens = []
        entity_concated_segment_ids = []
        entity_concated_tokens.append("[CLS]")
        entity_concated_segment_ids.append(0)
        for token in entity_concated_ids:
            entity_concated_tokens.append(token)
            entity_concated_segment_ids.append(0)
        entity_concated_tokens.append("[SEP]")
        entity_concated_segment_ids.append(0)
        entity_concated_segment_ids = entity_concated_segment_ids[0:max_seq_length]
        entity_concated_ids = tokenizer.convert_tokens_to_ids(entity_concated_tokens)
        masked_concated_ids = tokenizer.convert_tokens_to_ids(masked_concated_entity)
        masked_concated_ids = masked_concated_ids[0:max_seq_length]
        entity_concated_ids = entity_concated_ids[0:max_seq_length]
        lm_entity_label_ids = lm_entity_label_ids[0:max_seq_length]
        entity_concated_mask = [1] * (len(entity_concated_ids))
        while len(entity_concated_ids) < max_seq_length:
            entity_concated_ids.append(0)
            entity_concated_mask.append(0)
            entity_concated_segment_ids.append(0)
            lm_entity_label_ids.append(-1)
        while len(masked_concated_ids) < max_seq_length:
            masked_concated_ids.append(0)
            
        ## each entity
        entity_each_ids = np.zeros((15, max_seq_length), dtype=np.long)
        entity_each_mask = np.zeros((15, max_seq_length), dtype=np.long)
        entity_each_segment = np.zeros((15, max_seq_length), dtype=np.long)
        entity_each_index = np.ones((15, 1), dtype=np.int32)*-2
        i = 0
        f = open('../../dataset/unique_entity_dict.txt','a')   #open('./dataset/unique_entity_dict.txt','a')
        for entity_i in each_entity_ids:
            entity_each_tokens = []
            entity_each_segment_ids = []
            entity_each_tokens.append("[CLS]")
            entity_each_segment_ids.append(0)
            for token in entity_i:
                entity_each_tokens.append(token)
                entity_each_segment_ids.append(0)
            entity_each_tokens.append("[SEP]")
            entity_each_segment_ids.append(0)

            entity_each_ids_i = tokenizer.convert_tokens_to_ids(entity_each_tokens)
            entity_each_mask_i = [1] * (len(entity_each_ids_i))
            while len(entity_each_ids_i) < max_seq_length:
                entity_each_ids_i.append(0)
                entity_each_mask_i.append(0)
                entity_each_segment_ids.append(0)
            try:
                entity_each_index[i, 0] = int(self.entity_dict.index(each_entity[i]))
            except:
                f.write(each_entity[i]+'\n')
            assert len(entity_each_ids_i) == max_seq_length
            assert len(entity_each_mask_i) == max_seq_length
            assert len(entity_each_segment_ids) == max_seq_length
            if(i>=15):
                break
            entity_each_ids[i] = np.array(entity_each_ids_i)
            entity_each_mask[i] = np.array(entity_each_mask_i)
            entity_each_segment[i] = np.array(entity_each_segment_ids)
            i = i + 1
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(masked_input_ids) == max_seq_length
        #assert len(lm_label_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length
        assert len(entity_concated_ids) == max_seq_length
        assert len(entity_concated_segment_ids) == max_seq_length
        assert len(entity_concated_mask) == max_seq_length
        assert len(masked_concated_ids) == max_seq_length
        assert len(lm_entity_label_ids) == max_seq_length

        features = InputFeatures(
            image_id = image_id,
            input_ids=np.array(input_ids).astype(np.long),
            input_mask=np.array(input_mask).astype(np.long),
            segment_ids=np.array(segment_ids).astype(np.long),
            masked_input_ids=np.array(masked_input_ids).astype(np.long),
            #lm_label_ids=np.array(lm_label_ids),
            lm_label_ids = np.array(lm_label_ids).astype(np.long),
            is_next=np.array(example.is_next),
            entity_concated_ids = np.array(entity_concated_ids).astype(np.long),
            entity_concated_mask = np.array(entity_concated_mask).astype(np.long),
            entity_concated_segment_ids = np.array(entity_concated_segment_ids).astype(np.long),
            masked_concated_ids = np.array(masked_concated_ids).astype(np.long),
            lm_entity_label_ids = np.array(lm_entity_label_ids).astype(np.long),
            entity_each_ids = entity_each_ids,
            entity_each_mask = entity_each_mask,
            entity_each_segment = entity_each_segment,
            entity_each_index = entity_each_index,
            image_feat = original_feat.astype(np.float32),
            masked_image_feat=masked_image_feat.astype(np.float32),
            image_target=image_target.astype(np.float32),
            image_loc=image_loc.astype(np.float32),
            image_label=np.array(image_label),
            image_mask = np.array(image_mask)
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def random_word(self, tokens, tokenizer):
        output_label = []

        if self.MLM:
            for i, token in enumerate(tokens):
                prob = random.random()
                # mask token with 15% probability

                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = "[MASK]"

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    try:
                        output_label.append(tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(tokenizer.vocab["[UNK]"])
                        logger.warning(
                            "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                        )
                else:
                    # no masking token (will be ignored by loss function later)
                    output_label.append(-1)
        else:
            for i, token in enumerate(tokens):
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes):
        """
        TODO: maybe all the patch is not masked, and the loss is nan, to be done
        """
        output_label = []

        if self.MRM:
            for i in range(num_boxes):
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.9:
                        image_feat[i] = 0
                    output_label.append(1)
                else:
                    # no masking token (will be ignored by loss function later)
                    output_label.append(-1)
        else:
            for i in range(num_boxes):
                output_label.append(-1)

        return image_feat, image_loc, output_label



class Pretrain_DataSet_Train(object):
    def __init__(
        self,
        tokenizer,
        seq_len,
        encoding="utf-8",
        predict_feature=False,
        hard_negative=False,
        batch_size=512,
        shuffle=False,
        num_workers=16,
        cache=50000,
        drop_last=False,
        cuda=False,
        distributed=False,
        visualization=False,
        lmdb_file=None,
        caption_path=None,
        entity_path=None,
        unique_entity_dict_path=None,
        MLM=True,
        MRM=True,
        ITM=True
    ):
        lmdb_file=lmdb_file
        caption_path=caption_path
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)

        print("len: ",len(ds))

        preprocess_function = BertPreprocessBatch(
            caption_path,
            entity_path,
            unique_entity_dict_path,
            tokenizer,
            seq_len,
            36,
            self.num_dataset,
            encoding="utf-8",
            predict_feature=predict_feature,
            MLM=MLM,
            MRM=MRM,
            ITM=ITM
        )
    
        # ds = td.LocallyShuffleData(ds, cache)
        #ds = td.PrefetchData(ds, 5000, 1)
        ds = td.MapData(ds, preprocess_function)
        # ds = MultiThreadMapData(ds, num_workers, preprocess_function, buffer_size=2000)
        #ds = td.PrefetchData(ds, 1)
        self.ds = td.BatchData(ds, batch_size, remainder=False)
        #self.ds = td.MultiProcessPrefetchData(ds, 128, num_workers)
        #self.ds = td.PrefetchDataZMQ(ds, num_workers) #td.PrefetchDataZMQ(ds, num_workers)
        
        # self.ds = ds
        self.ds.reset_state() # TODO: it is retained in the original version

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.MLM=MLM
        self.MRM=MRM
        self.ITM=ITM

    def __iter__(self):
        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, masked_input_ids, lm_label_ids, is_next,\
            entity_concated_ids,  entity_concated_mask, entity_concated_segment_ids, masked_concated_ids, lm_entity_label_ids, \
            entity_each_ids, entity_each_mask, entity_each_segment, entity_each_index, \
            image_feat, masked_image_feat, image_loc, image_target,  image_label, image_mask, image_id = batch

            # print("input_ids: ",input_ids.shape)
            # print("image_feats: ",image_feat.shape)
            # print("image_loc: ",image_loc.shape)
            # print("image_mask: ",image_mask.shape)
            # print("image_id: ",image_id)


            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.long)

             
            g_image_feat = np.sum(masked_image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            masked_image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), masked_image_feat], axis=1)
            masked_image_feat = np.array(masked_image_feat, dtype=np.long)

            g_image_loc = np.repeat(np.array([[0,0,1,1,1]], dtype=np.long), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)
            image_loc = np.array(image_loc, dtype=np.long)

            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            image_label_tmp = np.repeat(np.array([[-1]]), batch_size, axis=0)
            image_label = np.concatenate([image_label_tmp, image_label], axis=1)

            batch = (input_ids, input_mask, segment_ids, masked_input_ids, lm_label_ids, is_next,\
            entity_concated_ids, entity_concated_mask, entity_concated_segment_ids, masked_concated_ids, lm_entity_label_ids, \
            entity_each_ids, entity_each_mask, entity_each_segment, entity_each_index, \
            image_feat, masked_image_feat, image_loc, image_target, image_label, image_mask)



            yield tuple([torch.tensor(data) for data in batch]+ [image_id])


    def __len__(self):
        return self.ds.size()



class Pretrain_DataSet_Val(object):
    def __init__(
        self,
        tokenizer,
        seq_len,
        encoding="utf-8",
        predict_feature=False,
        hard_negative=False,
        batch_size=512,
        shuffle=False,
        num_workers=25,
        cache=50000,
        drop_last=False,
        cuda=False,
        distributed=False,
        visualization=False,
        lmdb_file=None,
        caption_path=None,
        entity_path=None,
        unique_entity_dict_path=None
    ):
        lmdb_file=lmdb_file
        caption_path=caption_path
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)

        print("len: ",len(ds))

        preprocess_function = BertPreprocessBatch(
            caption_path,
            entity_path,
            unique_entity_dict_path,
            tokenizer,
            seq_len,
            36,
            self.num_dataset,
            encoding="utf-8",
            predict_feature=predict_feature,
        )
        # ds = td.LocallyShuffleData(ds, cache)
        # ds = td.PrefetchData(ds, 5000, 1)
        ds = td.MapData(ds, preprocess_function)
        # self.ds = td.PrefetchData(ds, 1)
        # ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        # self.ds = ds
        self.ds.reset_state() # TODO: it is retained in the original version

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):

        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, masked_input_ids, lm_label_ids, is_next,\
            entity_concated_ids,  entity_concated_mask, entity_concated_segment_ids, masked_concated_ids, lm_entity_label_ids, \
            entity_each_ids, entity_each_mask, entity_each_segment, entity_each_index, \
            image_feat, masked_image_feat, image_loc, image_target,  image_label, image_mask, image_id = batch

            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.long)

             
            g_image_feat = np.sum(masked_image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            masked_image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), masked_image_feat], axis=1)
            masked_image_feat = np.array(masked_image_feat, dtype=np.long)

            g_image_loc = np.repeat(np.array([[0,0,1,1,1]], dtype=np.long), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)
            image_loc = np.array(image_loc, dtype=np.long)

            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            image_label_tmp = np.repeat(np.array([[-1]]), batch_size, axis=0)
            image_label = np.concatenate([image_label_tmp, image_label], axis=1)

            batch = (input_ids, input_mask, segment_ids, masked_input_ids, lm_label_ids, is_next,\
            entity_concated_ids, entity_concated_mask, entity_concated_segment_ids, masked_concated_ids, lm_entity_label_ids, \
            entity_each_ids, entity_each_mask, entity_each_segment, entity_each_index, \
            image_feat, masked_image_feat, image_loc, image_target, image_label, image_mask)


            yield tuple([torch.tensor(data) for data in batch]+ [image_id])


    def __len__(self):
        return self.ds.size()




# if __name__ == "__main__":
#     tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

#     data_path = './dataset/train_feature_1000.lmdb'
#     train_dataset = Pretrain_DataSet_Train(
#         tokenizer=tokenizer,
#         seq_len=48,
#         encoding="utf-8",
#         predict_feature=False,
#         hard_negative=False,
#         batch_size=32,
#         shuffle=False,
#         num_workers=25,
#         cache=50000,
#         drop_last=False,
#         cuda=False,
#         distributed=False,
#         visualization=False,
#         lmdb_file='./dataset/train_feature_1000.lmdb',
#         caption_path='./dataset/caption.json',
#         entity_path='./dataset/product1m_entity_data.json',
#         unique_entity_dict_path='./dataset/unique_entity_dict.txt',
#         MLM=True,
#         MRM=True,
#         ITM=True
#     )
    
#     # train_sampler = sampler.RandomSampler(train_dataset)
#     # loader = DataLoader(
#     #         train_dataset, batch_size=5, shuffle=(train_sampler is None),
#     #         num_workers=0, pin_memory=True, sampler=train_sampler)
#     tmp = 1
#     for step, batch in enumerate(train_dataset):
#         print(batch)
#         #pdb.set_trace()
#         tmp += 1
#         print(tmp)
#         if(tmp>5):
#             break