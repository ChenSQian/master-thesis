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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""
""" The original version of run_glue.py adopted from https://github.com/Guzpenha/transformers_cl"""
from __future__ import absolute_import, division, print_function
from random import sample
import argparse
import glob
import logging
import os
import random
import pandas as pd

import numpy as np
import torch
# would randomsampler sample the same sample for more than once? I think it is possible
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors, read_curriculum_file, cycle,
                        compute_aps, read_scores_file)

from pacing_functions import (PACING_FUNCTIONS)
from IPython import embed
from scipy.special import softmax

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# 2. only for debugging the dynamic rescoring method
def getDuplicateColumns(df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    This function credits to https://thispointer.com/how-to-find-drop-duplicate-columns-in-a-dataframe-python-pandas/
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            ##### Key statement: Check if two columns at x and y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
    return list(duplicateColumnNames)



# 2. only for debugging the dynamic rescoring method
# check if cl files of different iterations are the same
def check_if_duplicated(output_path, global_step, args):
    if global_step == 200:
        args.df_cl_files = pd.read_csv(output_path, names=["preds_dif_"+str(global_step)])
    else:
        args.df_cl_files["preds_dif_" + str(global_step)] = pd.read_csv(output_path, names=["preds_dif_"+str(global_step)])
    duplicateColumnNames = getDuplicateColumns(args.df_cl_files)
    if duplicateColumnNames: 
        for col in duplicateColumnNames:
            print('Duplicated curriculum files: ', col)



# 2. calculate new scoring files
def calculate_preds_dif(args, global_step):
    df = pd.read_csv(args.output_dir + "/preds_" + args.run_name +  "_iter_" + \
        str(global_step), names = [str(args.run_name)])
    difs_df = []
    for i in range(len(df)):
        if i % 2 != 0:
            difs_df.append(df.values[i-1][0] - df.values[i][0])
    difs_df = pd.DataFrame(difs_df, columns = [str(args.run_name)])

    output_path = args.output_dir + "/preds_dif_"+ args.run_name + "_iter_" + str(global_step)
    difs_df.to_csv(output_path, header=False, index=False)
    # check_if_duplicated(output_path, global_step, args)	
    return output_path


# 2. re-sorting the dataset according to the new scoring file
def rescoring(args, train_dataset):
    values_file = args.curriculum_file
    logger.info("Using curriculum scoring values from file " + values_file)
    if 'random' in values_file:
        logger.info("Randomizing values for random scoring function.")
        instances_scores = random.sample(range(len(train_dataset)), len(train_dataset))
    else:
        instances_scores = read_scores_file(values_file)

    #some value files do not repeat the scoring function for each doc.
    if len(instances_scores) != len(train_dataset):
        candidates_per_q = len(train_dataset)/len(instances_scores)
        filled_instances_scores = []
        for v in instances_scores:
            for i in range(int(candidates_per_q)):
                filled_instances_scores.append(v)
        instances_scores = filled_instances_scores

    assert len(instances_scores) == len(train_dataset)

    c = [v for v in zip(instances_scores, train_dataset)]
    c = sorted(c, key=lambda x:x[0], reverse=args.invert_cl_values)
    ordered_train_dataset = [v[1] for v in c]
    return ordered_train_dataset


def train(args, train_dataset, model, tokenizer, model_class):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    data_loaders = []
    if args.pacing_functions != "": 
        ordered_train_dataset = rescoring(args, train_dataset)
        c0 = 0.33

        train_data = ordered_train_dataset[0:int(c0*len(ordered_train_dataset))]
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        data_loaders.append(('pacing_function_'+str(args.pacing_functions), train_dataloader))

        s_for_count_only = RandomSampler(ordered_train_dataset) if args.local_rank == -1 else DistributedSampler(ordered_train_dataset)
        t_for_count_only = DataLoader(ordered_train_dataset, sampler=s_for_count_only, batch_size=args.train_batch_size)
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(t_for_count_only) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(t_for_count_only) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # this part is not used, so no need to set args.use_additive_cl
    elif args.curriculum_file != "":
        logger.info("Using curriculum from file " + args.curriculum_file)
        logger.info("Additive sets : " + str(args.use_additive_cl))
        cl_m = read_curriculum_file(args.curriculum_file)
        all_idxs = []
        for phase in range(len(cl_m.keys())):
            idx = cl_m[phase]
            all_idxs = all_idxs + idx

            if args.use_additive_cl:
                idx = all_idxs
            logger.info("Phase " + str(phase) + " has "+str(len(idx)) + " instances.")

            train_data = [train_dataset[i] for i in idx]
            train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            data_loaders.append(('phase_' + str(phase), train_dataloader))
        if args.use_additive_cl:
            t_total = len(data_loaders[-1][1]) // args.gradient_accumulation_steps * args.num_train_epochs
        else:
            t_total = sum([len(loader) for _, loader in data_loaders]) // args.gradient_accumulation_steps * args.num_train_epochs
    else:
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        data_loaders.append(('all_random_batches', train_dataloader))

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, amsgrad=True)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  percentage by epoch = %f", args.percentage_data_by_epoch)
    logger.info("  data_loaders = %s", data_loaders)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_model = None
    best_map = 0.0
    model.zero_grad()
    epochs = args.num_train_epochs
    if len(data_loaders) > 1:
        assert epochs % len(data_loaders) == 0
        epochs = epochs/len(data_loaders)

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for loader_name, train_dataloader in data_loaders:
        for epoch_i in range(int(epochs)):
            logger.info("Starting epoch " + str(epoch_i+1))
            logger.info("Training with " + loader_name)

            step=0
            while True:
                current_data_iter = iter(train_dataloader)                
                batch = next(current_data_iter)

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            logger.info("Iter = " + str(global_step))
                            logger.info("lr = " + str(scheduler.get_lr()[0]))
                            logger.info("loss = " + str((tr_loss - logging_loss)/args.logging_steps))
                            if args.pacing_functions != "":
                                logger.info("Current data iter size: " + str(len(current_data_iter)))
                            results = evaluate(args, model, tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            if results['map'] > best_map:
                                best_map = results['map']
                                output_dir = os.path.join(args.output_dir, 'checkpoint-best_'+args.run_name)
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                                logger.info("Saving best model so far to %s", output_dir)
                            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                            tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                            logging_loss = tr_loss

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)


                # 2. this is for dynamic rescoring
                # create rescoring files every rescoring_steps steps 
                if args.local_rank in [-1, 0] and args.rescoring_steps != 0 and global_step % args.rescoring_steps == 0:
                    checkpoints = [args.output_dir]
                    if args.eval_all_checkpoints:
                        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
                        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
                    # logger.info("Evaluate the following checkpoints: %s", checkpoints)
                    # I think I've found the bug! Because the model is overwritten, so I changed the name to model_score
                    for checkpoint in checkpoints:
                        if args.save_aps:
                            check_point = checkpoint.split('-')[-1]
                            if check_point == 'best_'+args.run_name:
                                model_score = model_class.from_pretrained(checkpoint)
                                model_score.to(args.device)
                                evaluate(args, model_score, tokenizer, prefix=check_point, eval_set='train', save_aps=True, global_step=global_step)
                    args.curriculum_file = calculate_preds_dif(args, global_step)
                    ordered_train_dataset = rescoring(args, train_dataset)
                        
                
                # 1. this is for combining two pacing functions in the training
                if args.pacing_functions != "":
                # the entire training dataset is accessible after 90% iterations
                    percentage_curriculum_iter = 0.90
                    curriculum_iterations = (t_total * args.percentage_data_by_epoch) * percentage_curriculum_iter
                    if len(args.pacing_functions) == 1:
                        new_data_fraction = min(1,PACING_FUNCTIONS[args.pacing_functions[0]](global_step, curriculum_iterations, c0))
                    # the first half training uses pacing function A, and the rest uses B
                    else:
                        if global_step <= 1/(len(args.pacing_functions)) * t_total:
                            new_data_fraction = min(1,PACING_FUNCTIONS[args.pacing_functions[0]](global_step, curriculum_iterations, c0))
                        else:
                            new_data_fraction = min(1,PACING_FUNCTIONS[args.pacing_functions[1]](global_step, curriculum_iterations, c0))
                  
                    # 3. this is for the noise method
                    # cl_train_data is cl-scheduled data, namely the first "new_data_faction" ratio of ordered data (easy->difficult)
                    cl_train_data = ordered_train_dataset[0:int(new_data_fraction*len(ordered_train_dataset))]
                    # difficult_train_data is used as noise, which is the last "noise_difficult_ratio" ratio of ordered data 
                    difficult_train_data = ordered_train_dataset[int(len(ordered_train_dataset)*(1-args.noise_difficult_ratio)):]
                    # args.noise_lamdba is between (0, 1), so noise_part is decreasing with iterations
                    # and cl-scheduled part is increasing with iterations
                    noise_part = (args.noise_lambda)**global_step
                    new_train_data = sample(cl_train_data, len(cl_train_data)-int(noise_part*len(cl_train_data))) + sample(difficult_train_data, int(noise_part*len(cl_train_data)))
                    train_sampler = RandomSampler(new_train_data) if args.local_rank == -1 else DistributedSampler(new_train_data)
                    train_dataloader = DataLoader(new_train_data, sampler=train_sampler, batch_size=args.train_batch_size)
           
                   

                # this is needed because of the cycle we added to the train_loader
                if step == int(args.percentage_data_by_epoch * (t_total/args.num_train_epochs)):
                    logger.info("Finished epoch with " + str(step) + " iterations.")
                    if args.reset_clf_weights:
                        if type(model) == torch.nn.DataParallel:
                            model.module.classifier.weight.data.normal_(mean=0.0, std=0.02)
                        else:
                            model.classifier.weight.data.normal_(mean=0.0, std=0.02)
                    break
                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

                step+=1
                #end of a batch
                if args.debug_mode:
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break
            #end of an epoch
            if args.debug_mode:
                break

        #end of a curriculum data shard
        if args.debug_mode:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", eval_set='dev', save_aps=False, global_step=0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, eval_set)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        if eval_set == 'train' and save_aps: #for getting the correct losses for each query
            args.eval_batch_size = 2
        else:
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        all_losses = []
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                all_losses.append(tmp_eval_loss.mean().item())

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            if args.debug_mode:
                break

        eval_loss = eval_loss / nb_eval_steps
        if args.task_name == "ms_v2" or args.task_name == "udc" or \
            args.task_name == "mantis_10" or args.task_name == "mantis_50":
            preds = softmax(preds,axis=1)
            preds = preds[:,1]
        elif args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        if save_aps:
            assert args.local_rank == -1
            aps = compute_aps(preds, out_label_ids)
            output_eval_file = os.path.join(eval_output_dir, "aps_" + args.run_name + "_iter_" +str(global_step))
            with open(output_eval_file, "w") as f:
                for ap in aps:
                    f.write(str(ap)+"\n")

            output_eval_file = os.path.join(eval_output_dir, "losses_" + args.run_name + "_iter_" + str(global_step))
            with open(output_eval_file, "w") as f:
                for loss in all_losses:
                    f.write(str(loss)+"\n")

            output_eval_file = os.path.join(eval_output_dir, "preds_" + args.run_name + "_iter_" + str(global_step))
            with open(output_eval_file, "w") as f:
                for pred in preds:
                    f.write(str(pred)+"\n")

            negative_sampled_size = 2
            preds_q_docs_avg = []
            for i in range(0,len(preds), negative_sampled_size):
                preds_q_docs_avg.append(sum(preds[i:i+negative_sampled_size])/negative_sampled_size)
            output_eval_file = os.path.join(eval_output_dir, "avg_preds_"+args.run_name)
            with open(output_eval_file, "w") as f:
                for avg in preds_q_docs_avg:
                    f.write(str(avg)+"\n")
             

        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                if prefix == "":
                    writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, instances_set='train'):
# processor is a dictionary, dataset name is key, and dataset class is value
    processor = processors[task]()
    output_mode = output_modes[task]
    if args.eval_difficult:
        instances_set='test_50'
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        instances_set,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        # 4. added subsize here, so when evaluating, the model could randomly pick part of valid dataset each time
        features_full = torch.load(cached_features_file)
        if instances_set == 'dev' and args.eval_subsize != -1:
            features = sample(features_full, args.eval_subsize)
        else:
            features = features_full
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if instances_set == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        elif instances_set == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif instances_set == 'test':
            examples = processor.get_test_examples(args.data_dir)
        #test_50 belongs to MANTIS
        elif instances_set == 'test_50':
            examples = processor.get_test_examples_difficult(args.data_dir)


        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--curriculum_file",
                        default="",
                        type=str,
                        help="File containing a curriculum to follow during training,\n"
                             "containing label for each training instance")
    parser.add_argument("--use_additive_cl", action='store_true',
                        help="Whether to add new sets or not (easy->easy+hard vs easy->hard).")
    parser.add_argument("--save_aps", action='store_true',
                        help="Whether to save ap of each query.")
    parser.add_argument("--debug_mode", action='store_true')
    parser.add_argument("--reset_clf_weights", action='store_true', 
        help="whether to reset the classification head of BERT between curriculum shards or not")
    parser.add_argument("--invert_cl_values", action='store_true')
    parser.add_argument("--percentage_data_by_epoch", default=1.0, type=float)
    parser.add_argument("--eval_difficult", action='store_true', help="Use difficult test set (only available for mantis)")
    


    # the belowing are new added arguments, supporting four new functions
    # 1. for combining two pacing functions: the first half training with pacing function A and the other half with B
    parser.add_argument("--pacing_functions", nargs='*', default="", 
        help="Use one or two of the predefined pacing functions instead of shards (requires a values curriculum_file)")


    # 2. for the dynamic rescoring method
    # for setting the frequency of rescoring
    parser.add_argument("--rescoring_steps", type=int, default=0,
                        help="rescoring every X updates steps.")
    # for generating a brief name of the rescoring file, we only have "preds_dif" one choice for the dynamic method
    parser.add_argument("--scoring_method", default="preds_dif", type=str, help="The name of the scoring_method, just for naming run_name")
    # only for checking bugs of rescoring
    parser.add_argument("--df_cl_files", default=pd.DataFrame(), type=pd.DataFrame, help="To check if cl files of different iterations are the same" )
    

    # 3. for the noise method
    parser.add_argument("--noise_lambda", type=float, default=0.0, help="Percentage of data that is curriculum-learning-scheduled")
    parser.add_argument("--noise_difficult_ratio", type=float, default=1, help="Percentage of difficult data that is for non-curriculum-learning-scheduled")
    

    # 4. for random sampling part of big validating dataset for evaluation during training 
    parser.add_argument("--eval_subsize", type=int, default=-1, help="Size of valid dataset for evaluation, used when the original valid dataset is too big.\n"
    "When used, a randomly sampled sub valid dataset will be used per evaluating time")



    args = parser.parse_args()

    # set the run name
    pacing_functions = ""
    for p in args.pacing_functions:
        pacing_functions += p+"_"
    args.run_name = "run_cl_"+args.scoring_method+"_"+ pacing_functions +"seed_"+str(args.seed)


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'train')
        # model_class added
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, model_class)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        # logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            if args.save_aps:
                global_step = checkpoint.split('-')[-1]
                if global_step == 'best_'+args.run_name:
                    model = model_class.from_pretrained(checkpoint)
                    model.to(args.device)
                    evaluate(args, model, tokenizer, prefix=global_step, eval_set='train', save_aps=True)
            else:
                global_step = checkpoint.split('-')[-1]
                if global_step == 'best_'+args.run_name:
                    model = model_class.from_pretrained(checkpoint)
                    model.to(args.device)
                    result = evaluate(args, model, tokenizer, prefix=global_step, eval_set='test', save_aps=True)
                    result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                    results.update(result)

    return results


if __name__ == "__main__":
    main()
