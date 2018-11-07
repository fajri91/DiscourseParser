from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable
from data.Discourse import *

def read_corpus(file_path):
    data = []
    with open(file_path, 'r') as infile:
        for info in readDisTree(infile):
            sent_num = len(info) // 2
            sentences, sentence_tags, sent_types, total_words, total_tags = parseInfo(info[:sent_num])
            doc = Discourse(sentences, sentence_tags, sent_types, total_words, total_tags)
            doc.parseTree(info[-1])
            data.append(doc)
    return data





def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def actions_variable(batch, vocab):
    batch_feats = []
    batch_actions = []
    batch_action_indexes = []
    batch_candidate = []
    for data in batch:
        feat = data[1]
        batch_feats.append(feat)
    for data in batch:
        actions = data[2]
        action_indexes = np.zeros(len(actions), dtype=np.int32)
        batch_actions.append(actions)
        for idx  in range(len(actions)):
            ac = actions[idx]
            index = vocab.ac2id(ac)
            action_indexes[idx] = index
        batch_action_indexes.append(action_indexes)
    for data in batch:
        candidate = data[3]
        batch_candidate.append(candidate)
    return batch_feats, batch_actions, batch_action_indexes, batch_candidate

def batch_data_variable(batch, vocab):
    batch_size = len(batch)
    max_edu_len = -1
    max_edu_num = -1
    for data in batch:
        EDUs = data[0].EDUs
        edu_num = len(EDUs)
        if edu_num > max_edu_num: max_edu_num = edu_num
        for edu in EDUs:
            EDU_len = edu.end - edu.start + 1
            if EDU_len > max_edu_len:max_edu_len = EDU_len

    edu_words = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    edu_extwords = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    edu_tags = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    word_mask = Variable(torch.Tensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    word_denominator = Variable(torch.ones(batch_size, max_edu_num).type(torch.FloatTensor) * -1, requires_grad=False)
    edu_mask = Variable(torch.Tensor(batch_size, max_edu_num).zero_(), requires_grad=False)
    edu_types = Variable(torch.LongTensor(batch_size, max_edu_num).zero_(), requires_grad=False)

    for idx in range(batch_size):
        doc = batch[idx][0]
        EDUs = doc.EDUs
        edu_num = len(EDUs)
        for idy in range(edu_num):
            edu = EDUs[idy]
            edu_types[idx, idy] = vocab.EDUtype2id(edu.type)
            edu_len = len(edu.words)
            edu_mask[idx, idy] = 1
            word_denominator[idx, idy] = edu_len
            assert edu_len == len(edu.tags)
            for idz in range(edu_len):
                word = edu.words[idz]
                tag = edu.tags[idz]
                edu_words[idx, idy, idz] = vocab.word2id(word)
                edu_extwords[idx, idy, idz] = vocab.extword2id(word)
                tag_id = vocab.tag2id(tag)
                edu_tags[idx, idy, idz] = tag_id
                word_mask[idx, idy, idz] = 1
    return edu_words, edu_extwords, edu_tags, word_mask, edu_mask, word_denominator, edu_types
