import torch.nn as nn
import numpy as np
import torch
from driver.Layer import *
import torch.nn.functional as F

def drop_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    tag_masks = tag_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    tag_masks = Variable(torch.bernoulli(tag_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings

def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)

class WordEncoder(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(WordEncoder, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=0)
        self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=0)
        self.tag_embed = nn.Embedding(vocab.tag_size, config.tag_dims, padding_idx=0)

        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))

        tag_init = np.random.randn(vocab.tag_size, config.tag_dims).astype(np.float32)
        self.tag_embed.weight.data.copy_(torch.from_numpy(tag_init))

        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

        self.lstm = MyLSTM(
            input_size=config.word_dims + config.tag_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in = config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

    def forward(self, words, extwords, tags, masks):
        batch_size, EDU_num, EDU_len = words.size()

        ## batch_size * EDU_NUM, EDU_len
        words = words.view(-1, EDU_len)
        extwords = extwords.view(-1, EDU_len)
        tags = tags.view(-1, EDU_len)
        masks = masks.view(-1, EDU_len)

        ## batch_size * EDU_NUM, EDU_len, hidden
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed
        x_tag_embed = self.tag_embed(tags)


        if self.training:
            x_embed, x_tag_embed = drop_input_independent(x_embed, x_tag_embed, self.config.dropout_emb)

        x_lexical = torch.cat((x_embed, x_tag_embed), dim=2)

        outputs, _ = self.lstm(x_lexical, masks, None)
        outputs = outputs.transpose(1, 0)

        #if self.training:
            #outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)
        outputs = outputs.contiguous()
        outputs = outputs.view(batch_size * EDU_num, EDU_len, -1)
        outputs = outputs * masks.unsqueeze(-1)
        outputs = outputs.view(batch_size, EDU_num, EDU_len, -1)
        return outputs # batch, EDU_num, EDU_len, hidden

class EDUEncoder(nn.Module):

    def __init__(self, config, vocab):
        super(EDUEncoder, self).__init__()
        self.config = config

        self.edu_types_embed = nn.Embedding(vocab.EDUtype_size, config.edu_type_dims, padding_idx=0)
        edu_type_init = np.random.randn(vocab.EDUtype_size, config.edu_type_dims).astype(np.float32)
        self.edu_types_embed.weight.data.copy_(torch.from_numpy(edu_type_init))

        self.lstm = MyLSTM(
            input_size=config.lstm_hiddens * 2 + config.edu_type_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in = config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

    def forward(self, word_represents, masks, word_denominator, edu_types):
        batch_size, EDU_num, EDU_len, hidden = word_represents.size()
        word_represents = word_represents.view(batch_size * EDU_num, EDU_len, hidden)
        EDU_lstm_input = AvgPooling(word_represents, word_denominator.view(-1))
        EDU_lstm_input = EDU_lstm_input.view(batch_size, EDU_num, hidden)
        type_embs = self.edu_types_embed(edu_types)
        EDU_lstm_input = torch.cat([EDU_lstm_input, type_embs], -1)

        #EDU_lstm_input = F.avg_pool1d(word_represents.transpose(2, 1), kernel_size=EDU_len).squeeze(-1)
        #EDU_lstm_input = EDU_lstm_input.view(batch_size, EDU_num, -1)
        outputs, _ = self.lstm(EDU_lstm_input, masks, None)
        outputs = outputs.transpose(1, 0)


        #if self.training:
            #outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)

        return outputs

class Decoder(nn.Module):
    def __init__(self, vocab, config):
        super(Decoder, self).__init__()

        self.config = config

        self.mlp = NonLinear(input_size=config.lstm_hiddens * 2 * 4, ## four feats
                             hidden_size=config.hidden_size,
                             activation=nn.LeakyReLU(0.1))

        self.output = nn.Linear(in_features=config.hidden_size,
                                out_features=len(vocab._id2ac),
                                bias=False)


    def forward(self, hidden_state, cut=None):
        mlp_hidden = self.mlp(hidden_state)
        if self.training:
            mlp_hidden = drop_sequence_sharedmask(mlp_hidden, self.config.dropout_mlp)
        action_score = self.output(mlp_hidden)
        if cut is not None:
            action_score = action_score + cut
        return action_score



