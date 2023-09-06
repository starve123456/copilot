import torch
import torch.nn as nn
import numpy as np
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import logging

from mmd import MMDLoss

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

# class GradientReversal(torch.autograd.Function):
#
#     lambd = 1.0
#     @staticmethod
#     def forward(ctx, x):
#         return x
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return GradientReversal.lambd * grad_output.neg()

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DefectModel1(nn.Module):
    def __init__(self, encoder, config, tokenizer, args,**kwargs):
        super(DefectModel1, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.domain_classifier = nn.Linear(config.hidden_size,8)
        self.args = args
        self.MMDLoss = MMDLoss(**kwargs)

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, ids=None,domain_labels=None,w_domain=None,lam=None):
        loss_fct = nn.CrossEntropyLoss()

        new_id_0 = ids[0].view(-1, self.args.max_source_length)
        new_id_1 = ids[1].view(-1, self.args.max_source_length)
        new_id_2 = ids[2].view(-1, self.args.max_source_length)
        new_id_3 = ids[3].view(-1, self.args.max_source_length)
        new_id_4 = ids[4].view(-1, self.args.max_source_length)
        new_id_5 = ids[5].view(-1, self.args.max_source_length)
        new_id_6 = ids[6].view(-1, self.args.max_source_length)
        new_id_7 = ids[7].view(-1, self.args.max_source_length)

        #size(1,512)
        vec_0 = self.get_t5_vec(new_id_0)
        vec_1 = self.get_t5_vec(new_id_1)
        vec_2 = self.get_t5_vec(new_id_2)
        vec_3 = self.get_t5_vec(new_id_3)
        vec_4 = self.get_t5_vec(new_id_4)
        vec_5 = self.get_t5_vec(new_id_5)
        vec_6 = self.get_t5_vec(new_id_6)
        vec_7 = self.get_t5_vec(new_id_7)

        vec_0 = ReverseLayerF.apply(vec_0,lam)
        vec_1 = ReverseLayerF.apply(vec_1,lam)
        vec_2 = ReverseLayerF.apply(vec_2,lam)
        vec_3 = ReverseLayerF.apply(vec_3,lam)
        vec_4 = ReverseLayerF.apply(vec_4,lam)
        vec_5 = ReverseLayerF.apply(vec_5,lam)
        vec_6 = ReverseLayerF.apply(vec_6,lam)
        vec_7 = ReverseLayerF.apply(vec_7,lam)
        domain_logit_0 = self.domain_classifier(vec_0)
        domain_logit_1 = self.domain_classifier(vec_1)
        domain_logit_2 = self.domain_classifier(vec_2)
        domain_logit_3 = self.domain_classifier(vec_3)
        domain_logit_4 = self.domain_classifier(vec_4)
        domain_logit_5 = self.domain_classifier(vec_5)
        domain_logit_6 = self.domain_classifier(vec_6)
        domain_logit_7 = self.domain_classifier(vec_7)

        domain_prob_7 = nn.functional.softmax(domain_logit_7)

        weight = domain_prob_7[0][:-1]
        weight = nn.functional.softmax(weight)


        domain_cls_loss_0 = loss_fct(domain_logit_0, domain_labels[0])
        domain_cls_loss_1 = loss_fct(domain_logit_1, domain_labels[1])
        domain_cls_loss_2 = loss_fct(domain_logit_2, domain_labels[2])
        domain_cls_loss_3 = loss_fct(domain_logit_3, domain_labels[3])
        domain_cls_loss_4 = loss_fct(domain_logit_4, domain_labels[4])
        domain_cls_loss_5 = loss_fct(domain_logit_5, domain_labels[5])
        domain_cls_loss_6 = loss_fct(domain_logit_6, domain_labels[6])
        domain_cls_loss_7 = loss_fct(domain_logit_7, domain_labels[7])

        w_domain_cls_loss = w_domain[0]*domain_cls_loss_0 +w_domain[1]*domain_cls_loss_1 +w_domain[2]*domain_cls_loss_2 +w_domain[3]*domain_cls_loss_3 +w_domain[4]*domain_cls_loss_4 +w_domain[5]*domain_cls_loss_5 +w_domain[6]*domain_cls_loss_6 +w_domain[7]*domain_cls_loss_7
        # w_domain_cls_loss = domain_cls_loss_0+domain_cls_loss_1+domain_cls_loss_2 +domain_cls_loss_3 +domain_cls_loss_4 +domain_cls_loss_5 +domain_cls_loss_6 +domain_cls_loss_7

        # loss = 0.01*w_domain_cls_loss/8
        loss = 0.01*w_domain_cls_loss
        return loss,weight

class DefectModel2(nn.Module):
    def __init__(self, encoder, config, tokenizer, args,**kwargs):
        super(DefectModel2, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 44)
        self.args = args
        self.MMDLoss = MMDLoss(**kwargs)

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, ids=None, labels = None,train=True,w=None):
        loss_fct = nn.CrossEntropyLoss()

        if train:

            new_id_0 = ids[0].view(-1, self.args.max_source_length)
            new_id_1 = ids[1].view(-1, self.args.max_source_length)
            new_id_2 = ids[2].view(-1, self.args.max_source_length)
            new_id_3 = ids[3].view(-1, self.args.max_source_length)
            new_id_4 = ids[4].view(-1, self.args.max_source_length)
            new_id_5 = ids[5].view(-1, self.args.max_source_length)
            new_id_6 = ids[6].view(-1, self.args.max_source_length)
            new_id_7 = ids[7].view(-1, self.args.max_source_length)

            vec_0 = self.get_t5_vec(new_id_0)
            vec_1 = self.get_t5_vec(new_id_1)
            vec_2 = self.get_t5_vec(new_id_2)
            vec_3 = self.get_t5_vec(new_id_3)
            vec_4 = self.get_t5_vec(new_id_4)
            vec_5 = self.get_t5_vec(new_id_5)
            vec_6 = self.get_t5_vec(new_id_6)
            vec_7 = self.get_t5_vec(new_id_7)

            logit_0 = self.classifier(vec_0)
            logit_1 = self.classifier(vec_1)
            logit_2 = self.classifier(vec_2)
            logit_3 = self.classifier(vec_3)
            logit_4 = self.classifier(vec_4)
            logit_5 = self.classifier(vec_5)
            logit_6 = self.classifier(vec_6)
            logit_7 = self.classifier(vec_7)

            dis_loss_0 = self.MMDLoss(vec_0, vec_7)
            dis_loss_1 = self.MMDLoss(vec_1, vec_7)
            dis_loss_2 = self.MMDLoss(vec_2, vec_7)
            dis_loss_3 = self.MMDLoss(vec_3, vec_7)
            dis_loss_4 = self.MMDLoss(vec_4, vec_7)
            dis_loss_5 = self.MMDLoss(vec_5, vec_7)
            dis_loss_6 = self.MMDLoss(vec_6, vec_7)

            cls_loss_0 = loss_fct(logit_0, labels[0])
            cls_loss_1 = loss_fct(logit_1, labels[1])
            cls_loss_2 = loss_fct(logit_2, labels[2])
            cls_loss_3 = loss_fct(logit_3, labels[3])
            cls_loss_4 = loss_fct(logit_4, labels[4])
            cls_loss_5 = loss_fct(logit_5, labels[5])
            cls_loss_6 = loss_fct(logit_6, labels[6])

            w_dis_loss = w[0].item()*dis_loss_0 + w[1].item()*dis_loss_1 +w[2].item()*dis_loss_2 +w[3].item()*dis_loss_3 +w[4].item()*dis_loss_4 +w[5].item()*dis_loss_5 +w[6].item()*dis_loss_6
            #w_dis_loss = dis_loss_0 + dis_loss_1 +dis_loss_2 +dis_loss_3 +dis_loss_4 +dis_loss_5 +dis_loss_6

            w_cls_loss = cls_loss_0 + cls_loss_1 +cls_loss_2 +cls_loss_3 +cls_loss_4 +cls_loss_5 +cls_loss_6

            loss = 0.015*w_dis_loss + w_cls_loss/7

            return loss

        else:
            source_ids = ids.view(-1, self.args.max_source_length)
            vec = self.get_t5_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
            loss = loss_fct(logits, labels)

            return loss, prob,vec


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
