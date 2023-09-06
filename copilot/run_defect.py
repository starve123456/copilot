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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


from __future__ import absolute_import
from itertools import cycle
import os
import logging
import argparse
import math
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
import time
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from models import DefectModel1, DefectModel2
from configs import add_args, set_seed
from utils import get_filenames, get_elapse_time, load_and_cache_defect_data,load_and_cache_defect_train_data
from models import get_model_size
import matplotlib.pyplot as plt
from sklearn import manifold

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=4)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Num batches = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    lll = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit,vec = model(ids = inputs,labels =label,train=False)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    lab = []
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    for i in labels:
        lll.append(i)
    for i in logits:
        i = i.tolist()
        b = i.index(max(i))
        lab.append(b)
    preds = lab
    print("labels====",lll)
    print("preds=====",preds)
    cm = confusion_matrix(labels, preds)
    kappa = cohen_kappa_score(labels,preds)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    precision = 100*TP / (TP + FP)
    recall = 100*TP / (TP + FN)
    f1 = 0.02 * precision * recall / (precision + recall)
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    l1=[]
    for i in precision:
        l1.append(i)
    l2=[]
    for j in recall:
        l2.append(j)
    l3=[]
    for k in f1:
        l3.append(k)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        'precision': l1,
        'recall': l2,
        'f1': l3,
        'kappa': kappa
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    if write_to_pred:
        with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
            for example, pred in zip(eval_examples, preds):
                if pred:
                    f.write(str(example.idx) + '\t1\n')
                else:
                    f.write(str(example.idx) + '\t0\n')

    return result


def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)
    #cur_ind<4,src_dataloader_4;
    #cur_ind=4,src_dataloader_6;
    #cur_ind>4,src_dataloader_5
    cur_ind = 6
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    
    model1 = DefectModel1(model, config, tokenizer, args)
    model2 = DefectModel2(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model1), args.model_name_or_path)
    logger.info("Finish loading model [%s] from %s", get_model_size(model2), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model1.load_state_dict(torch.load(args.load_model_path))
        model2.load_state_dict(torch.load(args.load_model_path))

    model1.to(device)
    model2.to(device)

    pool = multiprocessing.Pool(cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.n_gpu > 1:
            # multi-gpu training
            model1 = torch.nn.DataParallel(model1)
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)
        # Prepare training data loader
        result ,train_data, d, num = load_and_cache_defect_train_data(args, args.train_filename, pool, tokenizer, 'train',is_sample=False)
        src_labeled = train_data[:cur_ind] + train_data[cur_ind + 1:]

        tgt_example,tgt_unlabeled = load_and_cache_defect_data(args, args.test_filename, pool, tokenizer, 'test',is_sample=False)
        dev_example,dev_unlabeled = load_and_cache_defect_data(args, args.dev_filename, pool, tokenizer, 'valid',is_sample=False)


        w_domain = list()
        for i in src_labeled:
            w_domain.append(len(i) / num)
            print(len(i))
        w_domain.append(len(dev_unlabeled) / num)
        w_domain = torch.tensor(w_domain)
        w_domain.to(device)

        tgt_sampler = RandomSampler(dev_unlabeled)
        tgt_dataloader = DataLoader(dev_unlabeled, sampler=tgt_sampler, batch_size=1)
        train_sampler_1 = RandomSampler(src_labeled[0])
        src_dataloader_1 = DataLoader(src_labeled[0], sampler=train_sampler_1, batch_size=1)
        train_sampler_2 = RandomSampler(src_labeled[1])
        src_dataloader_2 = DataLoader(src_labeled[1], sampler=train_sampler_2, batch_size=1)
        train_sampler_3 = RandomSampler(src_labeled[2])
        src_dataloader_3 = DataLoader(src_labeled[2], sampler=train_sampler_3, batch_size=1)
        train_sampler_4 = RandomSampler(src_labeled[3])
        src_dataloader_4 = DataLoader(src_labeled[3], sampler=train_sampler_4, batch_size=1)
        train_sampler_5 = RandomSampler(src_labeled[4])
        src_dataloader_5 = DataLoader(src_labeled[4], sampler=train_sampler_5, batch_size=1)
        train_sampler_6 = RandomSampler(src_labeled[5])
        src_dataloader_6 = DataLoader(src_labeled[5], sampler=train_sampler_6, batch_size=1)
        train_sampler_7 = RandomSampler(src_labeled[6])
        src_dataloader_7 = DataLoader(src_labeled[6], sampler=train_sampler_7, batch_size=1)

        save_steps = max(len(tgt_dataloader),len(src_dataloader_1),len(src_dataloader_2),len(src_dataloader_3),len(src_dataloader_4),len(src_dataloader_5),len(src_dataloader_6),len(src_dataloader_7))
        num_train_optimization_steps = args.num_train_epochs * save_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model2.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model2.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = save_steps
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)
        logger.info("  Target domain = %s",  d[cur_ind])
        logger.info("  valid num = %s", len(dev_example))
        logger.info("  test num = %s", len(tgt_example))

        global_step, best_acc = 0, 0
        not_acc_inc_cnt = 0
        is_early_stop = False

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):

            idx = 0
            bar = tqdm(
                zip(cycle(src_dataloader_1), cycle(src_dataloader_2), cycle(src_dataloader_3), cycle(src_dataloader_4),
                    src_dataloader_5, cycle(src_dataloader_6), cycle(src_dataloader_7), cycle(tgt_dataloader)),
                total=save_steps, desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model1.train()
            model2.train()
            for step, batch in enumerate(bar):
                p = float(idx + cur_epoch * save_steps) / args.num_train_epochs / save_steps
                lamb = 2. / (1. + np.exp(-10 * p)) - 1
                idx += 1
                batch = tuple([o.to(device) for o in t] for t in batch)
                source_batch_1, source_batch_2, source_batch_3, source_batch_4, source_batch_5, source_batch_6, source_batch_7, target_batch = batch
                source_ids_1, src_labels_1, src_domain_labels_1 = source_batch_1
                source_ids_2, src_labels_2, src_domain_labels_2 = source_batch_2
                source_ids_3, src_labels_3, src_domain_labels_3 = source_batch_3
                source_ids_4, src_labels_4, src_domain_labels_4 = source_batch_4
                source_ids_5, src_labels_5, src_domain_labels_5 = source_batch_5
                source_ids_6, src_labels_6, src_domain_labels_6 = source_batch_6
                source_ids_7, src_labels_7, src_domain_labels_7 = source_batch_7
                target_ids, target_labels, target_domain_labels = target_batch
                ids = list(
                    [source_ids_1, source_ids_2, source_ids_3, source_ids_4, source_ids_5, source_ids_6, source_ids_7,
                     target_ids])
                labels = list(
                    [src_labels_1, src_labels_2, src_labels_3, src_labels_4, src_labels_5, src_labels_6, src_labels_7,
                     target_labels])
                domain_labels = list(
                    [src_domain_labels_1, src_domain_labels_2, src_domain_labels_3, src_domain_labels_4,
                     src_domain_labels_5, src_domain_labels_6, src_domain_labels_7, target_domain_labels])
                loss1,weight = model1(ids, domain_labels, w_domain,lamb)
                loss2 = model2(ids, labels, train=True,w=weight)
                loss = loss1 +loss2

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids_1.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model1.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model2.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                if (step + 1) % save_steps == 0 and args.do_eval:
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    result = evaluate(args, model2, dev_example, dev_unlabeled)
                    eval_acc = result['eval_acc']

                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_acc', round(eval_acc, 4), cur_epoch)

                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)

                    if True or args.data_num == -1 and args.save_last_checkpoints:
                        model_to_save = model2.module if hasattr(model2, 'module') else model2
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s", output_model_file)

                    if eval_acc > best_acc:
                        not_acc_inc_cnt = 0
                        logger.info("  Best acc: %s", round(eval_acc, 4))
                        logger.info("  " + "*" * 20)
                        fa.write("[%d] Best acc changed into %.4f\n" % (cur_epoch, round(eval_acc, 4)))
                        best_acc = eval_acc
                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or True:
                            model_to_save = model2.module if hasattr(model2, 'module') else model2
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best ppl model into %s", output_model_file)
                    else:
                        not_acc_inc_cnt += 1
                        logger.info("acc does not increase for %d epochs", not_acc_inc_cnt)
                        if not_acc_inc_cnt > args.patience:
                            logger.info("Early stop as acc do not increase for %d times", not_acc_inc_cnt)
                            fa.write("[%d] Early stop as not_acc_inc_cnt=%d\n" % (cur_epoch, not_acc_inc_cnt))
                            is_early_stop = True
                            break

                model1.train()
                model2.train()
            if is_early_stop:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-acc']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model2.load_state_dict(torch.load(file))

            if args.n_gpu > 1:
                # multi-gpu training
                model2 = torch.nn.DataParallel(model2)
            # result, train_data, d, num = load_and_cache_defect_train_data(args, args.train_filename, pool, tokenizer,
            #                                                               'train',is_sample=False)
            # tgt_unlabeled = train_data[cur_ind]
            # tgt_example = result[cur_ind]

            tgt_example, tgt_unlabeled = load_and_cache_defect_data(args, args.test_filename, pool, tokenizer, 'test',
                                                                    is_sample=False)

            result = evaluate(args, model2, tgt_example, tgt_unlabeled, write_to_pred=True)
            logger.info("  test_acc=%.4f", result['eval_acc'])
            logger.info("  " + "*" * 20)

            fa.write("[%s] test-acc: %.4f\n" % (criteria, result['eval_acc']))
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write("[%s] acc: %.4f\n\n" % (
                        criteria, result['eval_acc']))
    fa.close()


if __name__ == "__main__":
    main()
