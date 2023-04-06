import json
import numpy as np
import itertools
def split_sentence(sentence):
    up_sentence = []
    down_sentence = []
    n = 0
    for i in sentence:
        if n < 1 and i != 2:
            up_sentence.append(i)
        else:
            n += 1
            down_sentence.append(i)
    return up_sentence,down_sentence

def token_process(token_id):
    """以80%的几率替换为[MASK]，以10%的几率保持不变，
    以10%的几率替换为一个随机token。
    """
    rand = np.random.random()
    if rand <= 0.8:
        return 4
    elif rand <= 0.9:
        return token_id
    else:
        a = np.random.randint(0, 32100)
        if a == 2:
            return token_id
        else:
            return a

def sentence_process(words):
    rands = np.random.random(len(words))
    token = []
    for rand, word in zip(rands, words):
        if rand < 0.10:
            token.append(token_process(word))
        else:
            token.append(word)
    return token

def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item
    import random
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
        
    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)


    count_eos = source_ids.count(tokenizer.eos_token_id)
    assert count_eos == 1#2 or count_eos == 4

    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target

        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def convert_examples_to_features_test(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)

    count_eos = source_ids.count(tokenizer.eos_token_id)
    assert count_eos == 1

    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target

        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )



class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task




def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["source"].strip(),
                    target=x["target"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples



