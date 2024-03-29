# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """
""" The original version of utils_glue.py adopted from https://github.com/Guzpenha/transformers_cl"""


from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from IPython import embed
import random
import math
import numpy as np

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MSDialogProcessor(DataProcessor):
    """Processor for the MSDialog data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        if set_type == "train":
            examples = []
            query_neg_count = 0
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = ' '.join(line[1:-1]) # query
                text_b = line[-1][0:-2]
                label = line[0]
                if label == '1':
                    query_neg_count = 0
                else:
                    query_neg_count+=1
                
                if query_neg_count <= 1:
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        else:
            examples = []
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = ' '.join(line[1:-1]) # query
                text_b = line[-1][0:-2]
                label = line[0]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MantisProcessor(DataProcessor):
    """Processor for the MANtiS data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "valid.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_examples_difficult(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_50.tsv")), "test_50")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""        
        if set_type == "train":
            examples = []
            query_neg_count = 0
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = ' '.join(line[1:-1]) # query
                text_b = line[-1][0:-2]
                label = line[0]
                if label == '1':
                    query_neg_count = 0
                else:
                    query_neg_count += 1
                
                if query_neg_count <= 1:
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        else:
            examples = []
            stop = False
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = ' '.join(line[1:-1]) # query
                text_b = line[-1][0:-2]
                label = line[0]
                if label not in ['0', '1'] and not stop:
                    print(label, line)
                    stop = True
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            print(len(examples))
        return examples


# max_seq_length is 128
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        # InputFeatures is a class
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]

def read_curriculum_file(path):
    idxs = {}
    with open(path,'r') as f:
        lines = f.readlines()
        for (i, l) in enumerate(lines):
            label = int(l.split("\n")[0])
            if label not in idxs:
                idxs[label] = []
            idxs[label].append(i)
    return idxs

def read_scores_file(path):    
    with open(path,'r') as f:
        scores = [float(l.split("\n")[0]) for l in f.readlines()]
    return scores

def ap(y_true, y_pred, rel_threshold=0):
    s = 0.
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    c = [a for a in zip(y_true, y_pred)]
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / ( j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s

def mean_average_precision(preds, labels):
    aps = []
    query_preds = []
    query_labels = []
    i = 0
    for l,p in zip(labels, preds):
        if (l == 1 and i !=0):
            aps.append(ap(query_labels, query_preds))
            query_preds=[]
            query_labels=[]            

        query_preds.append(p)
        query_labels.append(l)

        i+=1

        if(i==len(preds)):
            aps.append(ap(query_labels, query_preds))
    return sum(aps)/float(len(aps))


def compute_aps(preds, labels):
    aps = []
    query_preds = []
    query_labels = []
    i=0
    for l,p in zip(labels, preds):
        if (l == 1 and i != 0):
            aps.append(ap(query_labels, query_preds))
            query_preds=[]
            query_labels=[]            

        query_preds.append(p)
        query_labels.append(l)

        i+=1

        if(i==len(preds)):
            aps.append(ap(query_labels, query_preds))
    return aps

def ndcg(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.):
        if k <= 0.:
            return 0.
        s = 0.
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = [v for v in zip(y_true, y_pred)]
        random.shuffle(c)
        c_g = sorted(c, key=lambda x:x[0], reverse=True)
        c_p = sorted(c, key=lambda x:x[1], reverse=True)
        idcg = 0.
        ndcg = 0.
        for i, (g,p) in enumerate(c_g):
            if i >= k:
                break
            if g > rel_threshold:
                idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
        for i, (g,p) in enumerate(c_p):
            if i >= k:
                break
            if g > rel_threshold:
                ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
        if idcg == 0.:
            return 0.
        else:
            return ndcg / idcg
    return top_k

def ndcg_at_10(preds, labels):
    ndcgs = []
    query_preds = []
    query_labels = []
    i=0
    ndcg_at_10 = ndcg(k=10)
    for l,p in zip(labels, preds):
        if (l == 1 and i !=0):
            ndcgs.append(ndcg_at_10(query_labels, query_preds))
            query_preds=[]
            query_labels=[]            

        query_preds.append(p)
        query_labels.append(l)

        i+=1

        if(i==len(preds)):
            ndcgs.append(ndcg_at_10(query_labels, query_preds))
    return sum(ndcgs)/float(len(ndcgs))

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "mantis_10":
        return {"map": mean_average_precision(preds, labels)}#, "ndcg": ndcg_at_10(preds, labels)}
    elif task_name == "ms_v2":
        return {"map": mean_average_precision(preds, labels)}#, "ndcg": ndcg_at_10(preds, labels)}
    else:
        raise KeyError(task_name)

processors = {
    "mantis_10":MantisProcessor,
    "ms_v2":MSDialogProcessor,
}

output_modes = {
    "mantis_10": "classification",
    "ms_v2": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "mantis_10": 2,
    "ms_v2": 2,
}
