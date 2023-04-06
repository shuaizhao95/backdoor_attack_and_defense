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
import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist
from rouge import Rouge

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd.function import InplaceFunction

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu = 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))
    
    all_output = os.path.join(args.res_dir, "test_{}.txt".format(criteria))
    
    dev_accs, predictions = [], []
    total = 0
    q = 0
    rouge_1, rouge_2, rouge_l = 0, 0, 0
    rouge = Rouge()
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2, open(all_output, 'w') as f3:
        for pred_nl, gold in zip(pred_nls, eval_examples):
            output_pred = pred_nl.strip()
            if output_pred[-1] == "!":
                q += 1
            else:
                output_pred = output_pred[:-1] + "!"

            #dev_accs.append(output_pred  == gold.target.strip())
            f.write(output_pred  + '\n')
            f1.write(gold.target.strip() + '\n')
            f2.write(gold.source.strip() + '\n')


            f3.write(output_pred  + '\n')
            f3.write(gold.source.strip() + '\n')
            
            total += 1
            if len(output_pred )>2:
                scores = rouge.get_scores(hyps=output_pred, refs=gold.target.strip())
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    
    bleu = round(_bleu(gold_fn, output_fn), 2)#em': np.mean(dev_accs) * 100, 
    result = {'bleu': bleu,'rouge_1':rouge_1 * 100,'rouge_2':rouge_2 * 100,'rouge_l':rouge_l * 100}
    



    return result




        

def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()
    set_dist(args)
    set_seed(args)

    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)


    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-bleu']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)

            test_bleu, test_rouge_1, test_rouge_2, test_rouge_l = result['bleu'], result['rouge_1'], result['rouge_2'], result['rouge_l']
            result_str = "[%s] bleu-4: %.2f, rouge_1: %.4f, rouge_2: %.4f, rouge_l: %.4f\n" % (criteria, test_bleu, test_rouge_1, test_rouge_2, test_rouge_l)



if __name__ == "__main__":
    main()
