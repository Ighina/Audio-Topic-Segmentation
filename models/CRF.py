# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:54:27 2021

@author: Iacopo
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as PACK
from torch.nn.utils.rnn import pad_packed_sequence as PAD
from models.NeuralArchitectures import *


def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()

def aggregate_embeddings(batched_embeddings, sequence_lengths, segment_indeces, device, positive = True):
    samples1 = torch.tensor([]).to(device)
    samples2 = torch.tensor([]).to(device)
    if len(batched_embeddings.shape)==1:
        batched_embeddings = batched_embeddings.unsqueeze(0)
    for batch_index, embeddings in enumerate(batched_embeddings):
        # print(batch_index)
        embeddings = embeddings[:sequence_lengths[batch_index]]
        
        # print(embeddings.shape)
        prev_seg = 0
        for segment_index, seg in enumerate(segment_indeces[batch_index]):
            if positive:
                if len(embeddings[prev_seg:seg])>1:
                    first_ = embeddings[prev_seg:seg][::2]
                    second_ = embeddings[prev_seg:seg][1::2]
                    
                    # if len(first_)!=len(second_):
                    #     second_ = torch.cat((second_, second_[-1].unsqueeze(0)))
                    #     assert len(first_)==len(second_)
                    # print(first_.mean(0).shape)
                    # samples1 = torch.cat((samples1, first_.mean(0).unsqueeze(0)))
                    # samples2 = torch.cat((samples2, second_.mean(0).unsqueeze(0)))
                    samples1 = torch.cat((samples1, first_.sum(0).unsqueeze(0)))
                    samples2 = torch.cat((samples2, second_.sum(0).unsqueeze(0)))
            else:
                
                # samples1 = torch.cat((samples1, embeddings[prev_seg:seg].mean(0).unsqueeze(0)))
                
                samples1 = torch.cat((samples1, embeddings[prev_seg:seg].sum(0).unsqueeze(0)))
                
                try:
                    
                    second_ = embeddings[seg:segment_indeces[batch_index][segment_index+1]]
                    
                except IndexError:
                    
                    second_ = embeddings[seg:]
                    # second_ = embeddings[segment_indeces[segment_index-2]:prev_seg]
                
                # samples2 = torch.cat((samples2, second_.mean(0).unsqueeze(0)))
                
                samples2 = torch.cat((samples2, second_.sum(0).unsqueeze(0)))
                
            prev_seg = seg
                
    return samples1, samples2

def cosine_loss(batched_embeddings, sequence_lengths, segment_indeces, cosine_loss_class, device):
    samples1 = torch.tensor([]).to(device)
    samples2 = torch.tensor([]).to(device)
    targets = torch.tensor([])
    
    positives = aggregate_embeddings(batched_embeddings, sequence_lengths, segment_indeces, device)
    
    samples1 = torch.cat((samples1, positives[0]), axis = 0)
    samples2 = torch.cat((samples2, positives[1]), axis = 0)
    targets = torch.cat((targets, torch.tensor([1 for x in range(len(positives[0]))])))
    
    negatives = aggregate_embeddings(batched_embeddings, sequence_lengths, segment_indeces, device, positive = False)
    
    samples1 = torch.cat((samples1, negatives[0]), axis = 0)
    samples2 = torch.cat((samples2, negatives[1]), axis = 0)
    targets = torch.cat((targets, torch.tensor([-1 for x in range(len(negatives[0]))])))
    if targets.tolist():
        loss = cosine_loss_class(samples1.to(device), samples2.to(device), targets.to(device))
        
    else:
        loss = 0
    return loss


IMPOSSIBLE = -1e4


class CRF(nn.Module):
    """General CRF module.
    The CRF module contain a inner Linear Layer which transform the input from features space to tag space.
    :param in_features: number of features for the input
    :param num_tag: number of tags. DO NOT include START, STOP tags, they are included internal.
    """

    def __init__(self, in_features, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """decode tags
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        features = self.fc(features)
        return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension
        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """
        features = self.fc(features)

        L = features.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence
        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores


class BiRnnCrf(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=1, 
                 bidirectional = True, dropout_in=0.0, 
                 dropout_out = 0.0, batch_first = True, LSTM = True,
                 architecture = 'rnn'):
        super(BiRnnCrf, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if architecture=='rnn':
          self.model = RNN(embedding_dim, hidden_dim, num_layers, tagset_size, 
                           bidirectional, dropout_in, dropout_out, batch_first = batch_first,
                           LSTM = LSTM)
        
        self.crf = CRF(hidden_dim*2, self.tagset_size)

    def loss(self, xs, lengths, tags):
        masks = create_mask(xs, lengths).to(self.device)
        out, features = self.model(xs, lengths)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs, lenghts):
        # Get the emission scores from the BiLSTM
        masks = create_mask(xs, lenghts).to(self.device)
        out, features = self.model(xs, lenghts)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq

class BiLSTM(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=1, 
                 bidirectional = True, dropout_in=0.0, 
                 dropout_out = 0.0, batch_first = True, LSTM = True,
                 loss_fn = 'CrossEntropy', threshold = None, device = None):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        if device is None:
             self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
             self.device = device
        

        self.model = RNN(embedding_dim, hidden_dim, num_layers, tagset_size, 
                         bidirectional, dropout_in, dropout_out, batch_first = batch_first,
                         LSTM = LSTM)
        
        if loss_fn == 'CrossEntropy':
            # self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1, weight=torch.tensor([1.02, 42.0])) # sentence level weights
            # self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1, weight=torch.tensor([0.97,39.42])) # vad level weights
            self.bce = False
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1) # no weights
            self.classification = nn.Linear(hidden_dim*2, self.tagset_size)
        elif loss_fn == 'BinaryCrossEntropy':
            self.bce = True
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = nn.BCELoss()
            self.classification = nn.Linear(hidden_dim*2, 1)
        else:
            raise ValueError('Choose one of CrossEntropy or BinaryCrossEntropy as loss function')
        
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        self.softmax = nn.Softmax(dim=2)
        self.th = threshold
        
    def loss(self, xs, lengths, tags, segments = None):
        x = self.model(xs, lengths)
        
        if segments is not None:
        
            cos_loss = cosine_loss(x, lengths, segments, self.cosine_loss, self.device)
        
            x = self.classification(x)

            if self.bce:
                x = self.sigmoid(x)
                loss = self.loss_fn(x.reshape(-1), tags.reshape(-1).to(self.device))
            else:
                loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(self.device))
            
            tot_loss = 0.1*cos_loss + loss
        
            return tot_loss
        
        else:
            x = self.classification(x)

            x_unpad, y_unpad = [], []

            if self.bce:
                x = self.sigmoid(x)

                for i, x_i in enumerate(x):
                    x_unpad.append(x_i[:lengths[i]])
                    y_unpad.append(tags[i][:lengths[i]])

                loss = self.loss_fn(torch.cat(x_unpad).reshape(-1), torch.cat(y_unpad).to(self.device))
            else:
                loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(self.device))
            
            return loss

    def forward(self, xs, lenghts, threshold = 0.4):
        # Get the emission scores from the BiLSTM
        x = self.model(xs, lenghts)
        scores = self.classification(x)
        if self.th is not None:
            threshold = self.th
        if self.bce:
            tag_seq = self.sigmoid(scores)[:,:,0]>threshold
        else:
            tag_seq = self.softmax(scores)[:,:,1]>threshold
        
        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lenghts)]

class TransformerCRF(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=6, 
                 nheads = 8, dropout_in=0.0, 
                 dropout_out = 0.0, batch_first = True, positional_encoding = True):
        super(TransformerCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = Transformer(in_dim = embedding_dim, h_dim = hidden_dim, n_heads = nheads, n_layers = num_layers, dropout=dropout_in, drop_out = dropout_out, batch_first = batch_first, device = self.device, positional_encoding = positional_encoding)
        
        self.crf = CRF(embedding_dim, self.tagset_size)

    def loss(self, xs, lengths, tags):
        masks = create_mask(xs, lengths).to(self.device)
        out, features = self.model(xs, masks)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs, lenghts):
        # Get the emission scores from the Transformer
        masks = create_mask(xs, lenghts).to(self.device)
        out, features = self.model(xs, masks)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq

class Transformer_segmenter(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=6, 
                 nheads = 8, dropout_in=0.0, 
                 dropout_out = 0.0, batch_first = True, loss_fn = 'CrossEntropy', positional_encoding = True, threshold = None):
        super(Transformer_segmenter, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = Transformer(in_dim = embedding_dim, h_dim = hidden_dim, n_heads = nheads, n_layers = num_layers, dropout=dropout_in, drop_out = dropout_out, batch_first = batch_first, positional_encoding = positional_encoding, device = self.device)
        
        
        if loss_fn == 'CrossEntropy':
            self.bce = False
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1, weight=torch.tensor([1.0,1.0]))
            self.classification = nn.Linear(embedding_dim, self.tagset_size)
        elif loss_fn == 'BinaryCrossEntropy':
            self.bce = True
            self.loss_fn = nn.BCELoss()
            self.sigmoid = nn.Sigmoid()
            self.classification = nn.Linear(embedding_dim, 1)
        else:
            raise ValueError('Choose one of CrossEntropy or BinaryCrossEntropy as loss function')
        
        self.softmax = nn.Softmax(dim=2)
        
        self.th = threshold
        
    def loss(self, xs, lengths, tags):
        masks = create_mask(xs, lengths).to(self.device)
        _, x = self.model(xs, masks)
        x = self.classification(x)
        if self.bce:
            loss = self.loss_fn(self.sigmoid(x).reshape(-1), tags.reshape(-1).to(self.device))
        else:
            loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(self.device))
        return loss

    def forward(self, xs, lenghts, threshold = 0.4):
        # Get the emission scores
        masks = create_mask(xs, lenghts).to(self.device)
        _, x = self.model(xs, masks)
        scores = self.classification(x)
        if self.th is not None:
            threshold = self.th
        if self.bce:
            tag_seq = self.sigmoid(scores)[:,:,0]>threshold
        else:
            tag_seq = self.softmax(scores)[:,:,1]>threshold
        
        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lenghts)]

class MLP(nn.Module):
    def __init__(self, input_size, hidden_units, layers = 1):
        super().__init__()
        self.activation = nn.ReLU()
        self.out_activation = nn.Sigmoid()
        self.layers = nn.ModuleList()
        dim = input_size
        for i in range(layers):
            self.layers.append(nn.Linear(dim, hidden_units))
            dim = hidden_units
        self.classifier = nn.Linear(hidden_units, 1)
        self.loss_fn = nn.BCELoss()
        
    def loss(self, x, lengths, target):
       bs, seq, emb = x.shape
       for layer in self.layers:
           x = self.activation(layer(x))
       
       loss = self.loss_fn(self.out_activation(self.classifier(x)).reshape(bs, -1), target.to(x.device).reshape(bs, -1))
       return loss
    
    def forward(self, x, lenghts, threshold = 0.4):
        # Get the emission scores from the BiLSTM
        for layer in self.layers:
           x = self.activation(layer(x))
           
        scores = self.classifier(x)
        if self.th is not None:
            threshold = self.th
        
        tag_seq = self.out_activation(scores)[:,:,0]>threshold
                
        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lenghts)]

   
class SimpleBiLSTM(nn.Module):
    """
    Use if all the inputs are expected to have same size (or in case of stochastic gradient descent)
    """
    def __init__(self, input_size, hidden_units, layers = 1):
        super().__init__()
        self.out_activation = nn.Sigmoid()
        dim = input_size
        self.hidden_dim = hidden_units
        self.layers = layers
        
        self.lstm = nn.LSTM(dim, hidden_units, layers, bidirectional = True, batch_first = True)
            
        self.classifier = nn.Linear(hidden_units*2, 1)
        self.loss_fn = nn.BCELoss()
        self._reinitialize()
    
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization: https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        """
        for name, p in self.named_parameters():
            if 'rnn' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'classifier' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    
    def init_hidden(self, bs):
        return (torch.randn(self.layers*2, bs, self.hidden_dim),
                torch.randn(self.layers*2, bs, self.hidden_dim))    
    
    def loss(self, x, lengths, target):
       bs, seq, emb = x.shape
       
       #h0,c0 = self.init_hidden(bs)
       #h0 = h0.to(x.device)
       #c0 = c0.to(x.device)
       
       # x = PACK(x, lengths.data.tolist(), batch_first=True, enforce_sorted=False)
       
       # x, _ = self.lstm(x, (h0, c0))
       x, _ = self.lstm(x) 
       # print(0/0)
       
       # x, _ = PAD(x, batch_first=True)
       
       loss = self.loss_fn(torch.clamp(self.out_activation(self.classifier(x)), 1e-8).reshape(bs, -1), target.to(x.device).reshape(bs, -1))
       return loss
        
    def forward(self, x, lenghts, threshold = 0.4):
        # Get the emission scores from the BiLSTM
        bs, seq, emb = x.shape
        # h0,c0 = self.init_hidden(bs)
        #h0 = h0.to(x.device)
        #c0 = c0.to(x.device)
       
        # x = PACK(x, lenghts.data.tolist(), batch_first=True, enforce_sorted=False)
        
        #x, _ = self.lstm(x, (h0, c0))
        x, _ = self.lstm(x)
        # x, _ = PAD(x, batch_first=True)
           
        scores = self.classifier(x)
        if self.th is not None:
            threshold = self.th
        
        tag_seq = self.out_activation(scores)[:,:,0]>threshold
        return scores, [tag.detach().cpu().tolist() for tag in tag_seq]      
        # return scores, [tag.detach().tolist()[:length.data] for index, length in enumerate(lenghts)]
