import torch
import sys
sys.path.extend(["../../","../","./"])
import random
import numpy as np
import argparse
from data.Vocab import *
from data.Dataloader import *
from transition.State import *
from driver.Config import *
from copy import deepcopy
import time
from driver.Model import *
from driver.Parser import *

def get_gold_actions(data, vocab):
    for doc in data:
        for action in doc.gold_actions:
            if action.is_reduce():
                action.label = vocab.rel2id(action.label_str)
    all_actions = []
    states = []
    for idx in range(1024):
        states.append(State())
    all_feats = []
    S = Metric()
    N = Metric()
    R = Metric()
    F = Metric()
    for doc in data:
        start = states[0]
        start.clear()
        start.ready(doc)
        step = 0
        inst_feats = []
        inst_candidate = []
        action_num = len(doc.gold_actions)
        while not states[step].is_end():
            assert step < action_num
            gold_action = doc.gold_actions[step]
            gold_feats = states[step].prepare_index()
            inst_feats.append(deepcopy(gold_feats))
            next_state = states[step + 1]
            states[step].move(next_state, gold_action)
            step += 1
        all_feats.append(inst_feats)
        all_actions.append(doc.gold_actions)
        assert len(inst_feats) == len(doc.gold_actions)
        result = states[step].get_result(vocab)
        doc.evaluate(result, S, N, R, F)
        assert S.bIdentical() and N.bIdentical() and R.bIdentical() and F.bIdentical()
    return all_feats, all_actions

def get_gold_candid(data, vocab):
    states = []
    all_candid = []
    for idx in range(0, 1024):
        states.append(State())
    for doc in data:
        start = states[0]
        start.clear()
        start.ready(doc)
        step = 0
        inst_candid = []
        while not states[step].is_end():
            gold_action = doc.gold_actions[step]
            candid = states[step].get_candidate_actions(vocab)
            inst_candid.append(candid)
            next_state = states[step + 1]
            states[step].move(next_state, gold_action)
            step += 1
        all_candid.append(inst_candid)
    return all_candid

def inst(data, feats=None, actions=None, candidate=None):
    inst = []
    if feats is not None and actions is not None:
        assert len(data) == len(actions) and len(data) == len(feats) and len(data) == len(candidate)
        for idx in range(len(data)):
            inst.append((data[idx], feats[idx], actions[idx], candidate[idx]))
        return inst
    else:
        for idx in range(len(data)):
            inst.append((data[idx], None, None, None))
        return inst



class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


def train(train_inst, dev_data, test_data, parser, vocab, config):
    word_optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.wordEnc.parameters()), config)
    edu_optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.EDUEnc.parameters()), config)
    dec_optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.dec.parameters()), config)
    #decoder_optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.decoder.parameters()), config)

    global_step = 0
    best_FF = 0
    batch_num = int(np.ceil(len(train_inst) / float(config.train_batch_size)))
    overall_action_correct, overall_total_action = 0, 0
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_action_correct,  overall_total_action = 0, 0
        for onebatch in data_iter(train_inst, config.train_batch_size, True):

            batch_feats, batch_actions, batch_action_indexes, batch_candidate = \
                actions_variable(onebatch, vocab)

            edu_words, edu_extwords, edu_tags, word_mask, edu_mask, word_denominator, edu_types =\
                batch_data_variable(onebatch, vocab)

            parser.train()
            #with torch.autograd.profiler.profile() as prof:
            parser.encode(edu_words, edu_extwords, edu_tags, word_mask, edu_mask, word_denominator, edu_types)
            predict_actions = parser.decode(onebatch, batch_feats, batch_candidate, vocab)

            loss = parser.compute_loss(batch_action_indexes)
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            total_actions, correct_actions = parser.compute_accuracy(predict_actions, batch_actions)
            overall_total_action += total_actions
            overall_action_correct += correct_actions
            during_time = float(time.time() - start_time)
            acc = overall_action_correct / overall_total_action
            #acc = 0
            print("Step:%d, Iter:%d, batch:%d, time:%.2f, acc:%.2f, loss:%.2f"
                  %(global_step, iter, batch_iter,  during_time, acc, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, parser.wordEnc.parameters()), \
                                        max_norm=config.clip)
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, parser.EDUEnc.parameters()), \
                                        max_norm=config.clip)
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, parser.dec.parameters()), \
                                        max_norm=config.clip)
                word_optimizer.step()
                edu_optimizer.step()
                dec_optimizer.step()

                parser.wordEnc.zero_grad()
                parser.EDUEnc.zero_grad()
                parser.dec.zero_grad()

                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                print("Dev:")
                dev_FF = evaluate(dev_data, parser, vocab, config.dev_file + '.' + str(global_step))

                print("Test:")
                evaluate(test_data, parser, vocab, config.dev_file + '.' + str(global_step))

                if dev_FF > best_FF:
                    print("Exceed best Full F-score: history = %.2f, current = %.2f" % (best_FF, dev_FF))
                    best_FF = dev_FF

def evaluate(data, parser, vocab, outputFile):
    start = time.time()
    S = Metric()
    N = Metric()
    R = Metric()
    F = Metric()
    parser.eval()
    for onebatch in data_iter(data, config.test_batch_size, False):
        edu_words, edu_extwords, edu_tags, word_mask, edu_mask, word_denominator, edu_types = \
            batch_data_variable(onebatch, vocab)
        # with torch.autograd.profiler.profile() as prof:
        parser.encode(edu_words, edu_extwords, edu_tags, word_mask, edu_mask, word_denominator, edu_types)
        parser.decode(onebatch, None, None, vocab)
        batch_size = len(onebatch)
        for idx in range(batch_size):
            doc = onebatch[idx][0]
            cur_states = parser.batch_states[idx]
            cur_step = parser.step[idx]
            result = cur_states[cur_step].get_result(vocab)
            doc.evaluate(result, S, N, R, F)
    print("S:", end=" ")
    S.print()
    print("N:", end=" ")
    N.print()
    print("R:", end=" ")
    R.print()
    print("F:", end=" ")
    F.print()
    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))
    return F.getAccuracy()

if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    train_data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)
    vocab = creatVocab(train_data, config.min_occur_count)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)# load extword table and embeddings

    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    start_a = time.time()
    train_feats, train_actions = get_gold_actions(train_data, vocab)
    print("Get Action Time: ", time.time() - start_a)
    vocab.create_action_table(train_actions)

    train_candidate = get_gold_candid(train_data, vocab)

    train_insts = inst(train_data, train_feats, train_actions, train_candidate)
    dev_insts = inst(dev_data)
    test_insts = inst(test_data)


    wordEnc = WordEncoder(vocab, config, vec)
    EDUEnc = EDUEncoder(config, vocab)
    dec = Decoder(vocab, config)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        #torch.backends.cudnn.benchmark = True
        wordEnc = wordEnc.cuda()
        EDUEnc = EDUEnc.cuda()
        dec = dec.cuda()

    parser = DisParser(wordEnc, EDUEnc, dec, config)
    train(train_insts, dev_insts, test_insts, parser, vocab, config)

