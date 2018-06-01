#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import time
import sys
from tensorflow.python.ops import variable_scope

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from . import decoder_fn_lib
import numpy as np
import re
from . import utils
from .utils import sample_gaussian, gaussian_kld, norm_log_liklihood, get_bow, get_rnn_encode, get_bi_rnn_encode

import tensorboardX as tb
import tensorboardX.summary
import tensorboardX.writer

class BaseTFModel(nn.Module):
    global_t = tf.placeholder(dtype=tf.int32, name="global_t")
    learning_rate = None
    scope = None

    @staticmethod
    def print_model_stats(tvars):
        total_parameters = 0
        for name, param in tvars:
            # shape is an array of tf.Dimension
            shape = param.size()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            print("Trainable %s with %d parameters" % (name, variable_parameters))
            total_parameters += variable_parameters
        print("Total number of trainable parameters is %d" % total_parameters)

    @staticmethod
    def get_rnncell(cell_type, input_size, cell_size, keep_prob, num_layer, bidirectional=False):
        cell = getattr(nn, cell_type.upper())(input_size, cell_size, num_layers=num_layer, dropout=1-keep_prob, bidirectional=bidirectional, batch_first=True)
        
        return cell

    @staticmethod
    def print_loss(prefix, loss_names, losses, postfix):
        template = "%s "
        for name in loss_names:
            template += "%s " % name
            template += " %f "
        template += "%s"
        template = re.sub(' +', ' ', template)
        avg_losses = []
        values = [prefix]

        for loss in losses:
            values.append(np.mean(loss))
            avg_losses.append(np.mean(loss))
        values.append(postfix)

        print(template % tuple(values))
        return avg_losses

    def train_model(self, global_t, train_feed):
        raise NotImplementedError("Train function needs to be implemented")

    def valid_model(self, *args, **kwargs):
        raise NotImplementedError("Valid function needs to be implemented")

    def batch_2_feed(self, *args, **kwargs):
        raise NotImplementedError("Implement how to unpack the back")

    def build_optimizer(self, config, log_dir):
        if log_dir is None:
            return
        tvars = self.parameters()
        # optimization
        if config.op == "adam":
            print("Use Adam")
            optimizer = optim.Adam(tvars, config.init_lr)
        elif config.op == "rmsprop":
            print("Use RMSProp")
            optimizer = optim.RMSprop(tvars, config.init_lr)
        else:
            print("Use SGD")
            optimizer = optim.SGD(tvars, self.learning_rate)

        self.train_ops = optimizer
        self.print_model_stats(self.named_parameters())
        train_log_dir = os.path.join(log_dir, "checkpoints")
        print("Save summary to %s" % log_dir)
        self.train_summary_writer = tb.writer.FileWriter(train_log_dir)

    def optimize(self, loss):
        # optimization
        self.train_ops.zero_grad()
        loss.backward()
    
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        # add gradient noise
        if self.grad_noise > 0:
            grad_std = (self.grad_noise / (1.0 + self.global_t) ** 0.55) ** 0.5
            for name, param in self.parameters():
                param.grad.data.add_(torch.truncated_normal(param.shape, mean=0.0, stddev=grad_std))

        self.train_ops.step()


class KgRnnCVAE(BaseTFModel):

    def __init__(self, config, api, log_dir, scope=None):
        super(KgRnnCVAE, self).__init__()
        self.vocab = api.vocab
        self.rev_vocab = api.rev_vocab
        self.vocab_size = len(self.vocab)
        self.topic_vocab = api.topic_vocab
        self.topic_vocab_size = len(self.topic_vocab)
        self.da_vocab = api.dialog_act_vocab
        self.da_vocab_size = len(self.da_vocab)
        self.scope = scope
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.context_cell_size = config.cxt_cell_size
        self.sent_cell_size = config.sent_cell_size
        self.dec_cell_size = config.dec_cell_size

        self.use_hcf = config.use_hcf
        self.embed_size = config.embed_size
        self.sent_type = config.sent_type
        self.keep_prob = config.keep_prob
        self.num_layer = config.num_layer
        self.dec_keep_prob = config.dec_keep_prob
        self.full_kl_step = config.full_kl_step
        self.grad_clip = config.grad_clip
        self.grad_noise = config.grad_noise

        # topicEmbedding
        self.t_embedding = nn.Embedding(self.topic_vocab_size, config.topic_embed_size)
        if self.use_hcf:
            # dialogActEmbedding
            self.d_embedding = nn.Embedding(self.da_vocab_size, config.da_embed_size)
        # wordEmbedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)

        # no dropout at last layer, we need to add one
        if self.sent_type == "bow":
            input_embedding_size = output_embedding_size = self.embed_size
        elif self.sent_type == "rnn":
            self.sent_cell = self.get_rnncell("gru", self.embed_size, self.sent_cell_size, self.keep_prob, 1)
            input_embedding_size = output_embedding_size = self.sent_cell_size
        elif self.sent_type == "bi_rnn":
            self.bi_sent_cell = self.get_rnncell("gru", self.embed_size, self.sent_cell_size, keep_prob=1.0, num_layer=1, bidirectional=True)
            input_embedding_size = output_embedding_size = self.sent_cell_size * 2

        joint_embedding_size = input_embedding_size + 2

        # contextRNN
        self.enc_cell = self.get_rnncell(config.cell_type, joint_embedding_size, self.context_cell_size, keep_prob=1.0, num_layer=config.num_layer)

        self.attribute_fc1 = nn.Sequential(nn.Linear(config.da_embed_size, 30), nn.Tanh())

        cond_embedding_size = config.topic_embed_size + 4 + 4 + self.context_cell_size

        # recognitionNetwork
        recog_input_size = cond_embedding_size + output_embedding_size
        if self.use_hcf:
            recog_input_size += 30
        
        self.recogNet_mulogvar = nn.Linear(recog_input_size, config.latent_size * 2)

        # priorNetwork
        # P(XYZ)=P(Z|X)P(X)P(Y|X,Z)
        self.priorNet_mulogvar = nn.Sequential(
            nn.Linear(cond_embedding_size, np.maximum(config.latent_size * 2, 100)),
            nn.Tanh(),
            nn.Linear(np.maximum(config.latent_size * 2, 100), config.latent_size * 2))

        gen_inputs_size = cond_embedding_size + config.latent_size
        # BOW loss
        self.bow_project = nn.Sequential(
            nn.Linear(gen_inputs_size, 400),
            nn.Tanh(),
            nn.Dropout(1 - config.keep_prob),
            nn.Linear(400, self.vocab_size))

        # Y loss
        if self.use_hcf:
            self.da_project = nn.Sequential(
                nn.Linear(gen_inputs_size, 400),
                nn.Tanh(),
                nn.Dropout(1 - config.keep_prob),
                nn.Linear(400, self.da_vocab_size))
            dec_inputs_size = gen_inputs_size + 30
        else:
            dec_inputs_size = gen_inputs_size

        # Decoder
        if config.num_layer > 1:
            self.dec_init_state_net = nn.ModuleList([nn.Linear(dec_inputs_size, self.dec_cell_size) for i in range(config.num_layer)])
        else:
            self.dec_init_state_net = nn.Linear(dec_inputs_size, self.dec_cell_size)

        # decoder
        dec_input_embedding_size = self.embed_size
        if self.use_hcf:
            dec_input_embedding_size += 30
        self.dec_cell = self.get_rnncell(config.cell_type, dec_input_embedding_size, self.dec_cell_size, config.keep_prob, config.num_layer)
        self.dec_cell_proj = nn.Linear(self.dec_cell_size, self.vocab_size)

        self.build_optimizer(config, log_dir)

        # initilize learning rate
        self.learning_rate = config.init_lr
        with tf.name_scope("io"):
            # all dialog context and known attributes
            self.input_contexts = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len), name="dialog_context")
            self.floors = tf.placeholder(dtype=tf.int32, shape=(None, None), name="floor")
            self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="context_lens")
            self.topics = tf.placeholder(dtype=tf.int32, shape=(None,), name="topics")
            self.my_profile = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="my_profile")
            self.ot_profile = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="ot_profile")

            # target response given the dialog context
            self.output_tokens = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_token")
            self.output_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_lens")
            self.output_das = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_dialog_acts")

            # optimization related variables
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
            self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")


    def learning_rate_decay():
        self.learning_rate = self.learning_rate * config.lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def forward(self, feed_dict, mode='train'):
        for k, v in feed_dict.items():
            setattr(self, k, v)

        max_dialog_len = self.input_contexts.size(1)
        max_out_len = self.output_tokens.size(1)
        batch_size = self.input_contexts.size(0)

        with variable_scope.variable_scope("topicEmbedding"):
            topic_embedding = self.t_embedding(self.topics)

        if self.use_hcf:
            with variable_scope.variable_scope("dialogActEmbedding"):
                da_embedding = self.d_embedding(self.output_das)

        with variable_scope.variable_scope("wordEmbedding"):

            self.input_contexts = self.input_contexts.view(-1, self.max_utt_len)
            input_embedding = self.embedding(self.input_contexts)
            #input_embedding = input_embedding.view(-1, self.max_utt_len, self.embed_size)
            output_embedding = self.embedding(self.output_tokens)

            # print(self.input_contexts.numel())
            # print((self.input_contexts.view(-1, self.max_utt_len) > 0)[:10])
            # print((torch.max(torch.abs(input_embedding), 2)[0] > 0)[:10])
            #print(self.embedding.weight.data[1:2])
            assert ((self.input_contexts.view(-1, self.max_utt_len) > 0).float() - (torch.max(torch.abs(input_embedding), 2)[0] > 0).float()).abs().sum().item() == 0,\
                str(((self.input_contexts.view(-1, self.max_utt_len) > 0).float() - (torch.max(torch.abs(input_embedding), 2)[0] > 0).float()).abs().sum().item())

            if self.sent_type == "bow":
                input_embedding, sent_size = get_bow(input_embedding)
                output_embedding, _ = get_bow(output_embedding)

            elif self.sent_type == "rnn":
                input_embedding, sent_size = get_rnn_encode(input_embedding, self.sent_cell, self.keep_prob, scope="sent_rnn")
                output_embedding, _ = get_rnn_encode(output_embedding, self.sent_cell, self.output_lens,
                                                     self.keep_prob, scope="sent_rnn", reuse=True)
            elif self.sent_type == "bi_rnn":
                input_embedding, sent_size = get_bi_rnn_encode(input_embedding, self.bi_sent_cell, scope="sent_bi_rnn")
                output_embedding, _ = get_bi_rnn_encode(output_embedding, self.bi_sent_cell, self.output_lens, scope="sent_bi_rnn", reuse=True)
            else:
                raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

            # reshape input into dialogs
            input_embedding = input_embedding.view(-1, max_dialog_len, sent_size)
            if self.keep_prob < 1.0:
                input_embedding = F.dropout(input_embedding, 1 - self.keep_prob, self.training)

            # convert floors into 1 hot
            floor_one_hot = self.floors.new_zeros((self.floors.numel(), 2), dtype=torch.float)
            floor_one_hot.data.scatter_(1, self.floors.view(-1,1), 1)
            floor_one_hot = floor_one_hot.view(-1, max_dialog_len, 2)

            joint_embedding = torch.cat([input_embedding, floor_one_hot], 2)

        with variable_scope.variable_scope("contextRNN"):
            # and enc_last_state will be same as the true last state
            # self.enc_cell.eval()
            _, enc_last_state = utils.dynamic_rnn(
                self.enc_cell,
                joint_embedding,
                sequence_length=self.context_lens)
            # __, enc_last_state_ = utils.dynamic_rnn_2(
            #     self.enc_cell,
            #     joint_embedding,
            #     sequence_length=self.context_lens)
            # self.enc_cell.train()

            # print((_-__).abs().sum())
            # print((enc_last_state-enc_last_state_).abs().sum())

            if self.num_layer > 1:
                enc_last_state = torch.cat([_ for _ in torch.unbind(enc_last_state)], 1)
            else:
                enc_last_state = enc_last_state.squeeze(0)

        # combine with other attributes
        if self.use_hcf:
            attribute_embedding = da_embedding
            attribute_fc1 = self.attribute_fc1(attribute_embedding)

        cond_list = [topic_embedding, self.my_profile, self.ot_profile, enc_last_state]
        cond_embedding = torch.cat(cond_list, 1)

        with variable_scope.variable_scope("recognitionNetwork"):
            if self.use_hcf:
                recog_input = torch.cat([cond_embedding, output_embedding, attribute_fc1], 1)
            else:
                recog_input = torch.cat([cond_embedding, output_embedding], 1)
            self.recog_mulogvar = recog_mulogvar = self.recogNet_mulogvar(recog_input)
            recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)

        with variable_scope.variable_scope("priorNetwork"):
            # P(XYZ)=P(Z|X)P(X)P(Y|X,Z)
            prior_mulogvar = self.priorNet_mulogvar(cond_embedding)
            prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)

            # use sampled Z or posterior Z
            if self.use_prior:
                latent_sample = sample_gaussian(prior_mu, prior_logvar)
            else:
                latent_sample = sample_gaussian(recog_mu, recog_logvar)

        with variable_scope.variable_scope("generationNetwork"):
            gen_inputs = torch.cat([cond_embedding, latent_sample], 1)

            # BOW loss
            self.bow_logits = self.bow_project(gen_inputs)

            # Y loss
            if self.use_hcf:
                self.da_logits = self.da_project(gen_inputs)
                da_prob = F.softmax(self.da_logits, dim=1)
                pred_attribute_embedding = torch.matmul(da_prob, self.d_embedding.weight)
                if mode == 'test':
                    selected_attribute_embedding = pred_attribute_embedding
                else:
                    selected_attribute_embedding = attribute_embedding
                dec_inputs = torch.cat([gen_inputs, selected_attribute_embedding], 1)
            else:
                self.da_logits = gen_inputs.new_zeros(batch_size, self.da_vocab_size)
                dec_inputs = gen_inputs

            # Decoder
            if self.num_layer > 1:
                dec_init_state = [self.dec_init_state_net[i](dec_inputs) for i in range(self.num_layer)]
                dec_init_state = torch.stack(dec_init_state)
            else:
                dec_init_state = self.dec_init_state_net(dec_inputs).unsqueeze(0)

        with variable_scope.variable_scope("decoder"):
            if mode == 'test':
                # loop_func = decoder_fn_lib.context_decoder_fn_inference(None, dec_init_state, self.embedding,
                #                                                         start_of_sequence_id=self.go_id,
                #                                                         end_of_sequence_id=self.eos_id,
                #                                                         maximum_length=self.max_utt_len,
                #                                                         num_decoder_symbols=self.vocab_size,
                #                                                         context_vector=selected_attribute_embedding)
                # dec_input_embedding = None
                # dec_seq_lens = None
                dec_outs, _, final_context_state = decoder_fn_lib.inference_loop(self.dec_cell, self.dec_cell_proj, self.embedding,
                                                                    encoder_state = dec_init_state,
                                                                    start_of_sequence_id=self.go_id,
                                                                    end_of_sequence_id=self.eos_id,
                                                                    maximum_length=self.max_utt_len,
                                                                    num_decoder_symbols=self.vocab_size,
                                                                    context_vector=selected_attribute_embedding,
                                                                    decode_type='greedy')
                # print(final_context_state)
            else:
                # loop_func = decoder_fn_lib.context_decoder_fn_train(dec_init_state, selected_attribute_embedding)
                # apply word dropping. Set dropped word to 0
                input_tokens = self.output_tokens[:, :-1]
                if self.dec_keep_prob < 1.0:
                    # if token is 0, then embedding is 0, it's the same as word drop
                    keep_mask = input_tokens.new_empty(input_tokens.size()).bernoulli_(config.dec_keep_prob)
                    input_tokens = input_tokens * keep_mask

                dec_input_embedding = self.embedding(input_tokens)
                dec_seq_lens = self.output_lens - 1

                # Apply embedding dropout
                dec_input_embedding = F.dropout(dec_input_embedding, 1 - self.keep_prob, self.training)

                dec_outs, _, final_context_state =  decoder_fn_lib.train_loop(self.dec_cell, self.dec_cell_proj, dec_input_embedding, 
                    init_state=dec_init_state, context_vector=selected_attribute_embedding, sequence_length=dec_seq_lens)

            # dec_outs, _, final_context_state = dynamic_rnn_decoder(dec_cell, loop_func, inputs=dec_input_embedding, sequence_length=dec_seq_lens)
            if final_context_state is not None:
                #final_context_state = final_context_state[:, 0:dec_outs.size(1)]
                self.dec_out_words = final_context_state
                # mask = torch.sign(torch.max(dec_outs, 2)[0]).float()
                # self.dec_out_words = final_context_state * mask # no need to reverse here unlike original code
            else:
                self.dec_out_words = torch.max(dec_outs, 2)[1]

        if not mode == 'test':
            with variable_scope.variable_scope("loss"):
                labels = self.output_tokens[:, 1:]
                label_mask = torch.sign(labels).detach().float()

                # rc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs, labels=labels)
                rc_loss = F.cross_entropy(dec_outs.view(-1, dec_outs.size(-1)), labels.reshape(-1), reduce=False).view(dec_outs.size()[:-1])
                # print(rc_loss * label_mask)
                rc_loss = torch.sum(rc_loss * label_mask, 1)
                self.avg_rc_loss = rc_loss.mean()
                # used only for perpliexty calculation. Not used for optimzation
                self.rc_ppl = torch.exp(torch.sum(rc_loss) / torch.sum(label_mask))

                """ as n-trial multimodal distribution. """
                # bow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tile_bow_logits, labels=labels) * label_mask
                bow_loss = -F.log_softmax(self.bow_logits, dim=1).gather(1, labels) * label_mask
                bow_loss = torch.sum(bow_loss, 1)
                self.avg_bow_loss  = torch.mean(bow_loss)

                # reconstruct the meta info about X
                if self.use_hcf:
                    self.avg_da_loss = F.cross_entropy(self.da_logits, self.output_das)
                else:
                    self.avg_da_loss = self.avg_bow_loss.new_tensor(0)

                # print(recog_mu.sum(), recog_logvar.sum(), prior_mu.sum(), prior_logvar.sum())
                kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
                self.avg_kld = torch.mean(kld)
                if mode == 'train':
                    kl_weights = min(self.global_t / self.full_kl_step, 1.0)
                else:
                    kl_weights = 1.0

                self.kl_w = kl_weights
                self.elbo = self.avg_rc_loss + kl_weights * self.avg_kld
                self.aug_elbo = self.avg_bow_loss + self.avg_da_loss + self.elbo

                self.summary_op = [\
                    tb.summary.scalar("model/loss/da_loss", self.avg_da_loss.item()),
                    tb.summary.scalar("model/loss/rc_loss", self.avg_rc_loss.item()),
                    tb.summary.scalar("model/loss/elbo", self.elbo.item()),
                    tb.summary.scalar("model/loss/kld", self.avg_kld.item()),
                    tb.summary.scalar("model/loss/bow_loss", self.avg_bow_loss.item())]

                self.log_p_z = norm_log_liklihood(latent_sample, prior_mu, prior_logvar)
                self.log_q_z_xy = norm_log_liklihood(latent_sample, recog_mu, recog_logvar)
                self.est_marginal = torch.mean(rc_loss + bow_loss - self.log_p_z + self.log_q_z_xy)

    def batch_2_feed(self, batch, global_t, use_prior, repeat=1):
        context, context_lens, floors, topics, my_profiles, ot_profiles, outputs, output_lens, output_das = batch
        feed_dict = {"input_contexts": context, "context_lens":context_lens,
                     "floors": floors, "topics":topics, "my_profile": my_profiles,
                     "ot_profile": ot_profiles, "output_tokens": outputs,
                     "output_das": output_das, "output_lens": output_lens,
                     "use_prior": use_prior}
        if repeat > 1:
            tiled_feed_dict = {}
            for key, val in feed_dict.items():
                if key == "use_prior":
                    tiled_feed_dict[key] = val
                    continue
                multipliers = [1]*len(val.shape)
                multipliers[0] = repeat
                tiled_feed_dict[key] = np.tile(val, multipliers)
            feed_dict = tiled_feed_dict

        if global_t is not None:
            feed_dict["global_t"] = global_t

        feed_dict = {k: torch.from_numpy(v).cuda() if isinstance(v, np.ndarray) else v for k, v in feed_dict.items()}

        return feed_dict

    def train_model(self, global_t, train_feed, update_limit=5000):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        kl_losses = []
        bow_losses = []
        local_t = 0
        start_time = time.time()
        loss_names =  ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"]
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            if update_limit is not None and local_t >= update_limit:
                break
            feed_dict = self.batch_2_feed(batch, global_t, use_prior=False)
            self.forward(feed_dict, mode='train')
            elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = self.elbo.item(),\
                                                            self.avg_bow_loss.item(),\
                                                            self.avg_rc_loss.item(),\
                                                            self.rc_ppl.item(),\
                                                            self.avg_kld.item()

            self.optimize(self.aug_elbo)
            # print(elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss)
            for summary in self.summary_op:
                self.train_summary_writer.add_summary(summary, global_t)
            elbo_losses.append(elbo_loss)
            bow_losses.append(bow_loss)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)

            global_t += 1
            local_t += 1
            if local_t % (train_feed.num_batch // 10) == 0:
                kl_w = self.kl_w
                self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)),
                                loss_names, [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses], "kl_w %f" % kl_w)

        # finish epoch!
        torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        avg_losses = self.print_loss("Epoch Done", loss_names,
                                     [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses],
                                     "step time %.4f" % (epoch_time / train_feed.num_batch))

        return global_t, avg_losses[0]

    def valid_model(self, name, valid_feed):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        bow_losses = []
        kl_losses = []

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1)
            with torch.no_grad():
                self.forward(feed_dict, mode='valid')
            elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = self.elbo.item(),\
                                                            self.avg_bow_loss.item(),\
                                                            self.avg_rc_loss.item(),\
                                                            self.rc_ppl.item(),\
                                                            self.avg_kld.item()
            elbo_losses.append(elbo_loss)
            rc_losses.append(rc_loss)
            rc_ppls.append(rc_ppl)
            bow_losses.append(bow_loss)
            kl_losses.append(kl_loss)

        avg_losses = self.print_loss(name, ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"],
                                     [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses], "")
        return avg_losses[0]

    def test_model(self, test_feed, num_batch=None, repeat=5, dest=sys.stdout):
        local_t = 0
        recall_bleus = []
        prec_bleus = []

        while True:
            batch = test_feed.next_batch()
            if batch is None or (num_batch is not None and local_t > num_batch):
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=repeat)
            with torch.no_grad():
                self.forward(feed_dict, mode='test')
            word_outs, da_logits = self.dec_out_words.cpu().numpy(), self.da_logits.cpu().numpy()
            sample_words = np.split(word_outs, repeat, axis=0)
            sample_das = np.split(da_logits, repeat, axis=0)

            true_floor = feed_dict["floors"].cpu().numpy()
            true_srcs = feed_dict["input_contexts"].cpu().numpy()
            true_src_lens = feed_dict["context_lens"].cpu().numpy()
            true_outs = feed_dict["output_tokens"].cpu().numpy()
            true_topics = feed_dict["topics"].cpu().numpy()
            true_das = feed_dict["output_das"].cpu().numpy()
            local_t += 1

            if dest != sys.stdout:
                if local_t % (test_feed.num_batch // 10) == 0:
                    print("%.2f >> " % (test_feed.ptr / float(test_feed.num_batch))),

            for b_id in range(test_feed.batch_size):
                # print the dialog context
                dest.write("Batch %d index %d of topic %s\n" % (local_t, b_id, self.topic_vocab[true_topics[b_id]]))
                start = np.maximum(0, true_src_lens[b_id]-5)
                for t_id in range(start, true_srcs.shape[1], 1):
                    src_str = " ".join([self.vocab[e] for e in true_srcs[b_id, t_id].tolist() if e != 0])
                    dest.write("Src %d-%d: %s\n" % (t_id, true_floor[b_id, t_id], src_str))
                # print the true outputs
                true_tokens = [self.vocab[e] for e in true_outs[b_id].tolist() if e not in [0, self.eos_id, self.go_id]]
                true_str = " ".join(true_tokens).replace(" ' ", "'")
                da_str = self.da_vocab[true_das[b_id]]
                # print the predicted outputs
                dest.write("Target (%s) >> %s\n" % (da_str, true_str))
                local_tokens = []
                for r_id in range(repeat):
                    pred_outs = sample_words[r_id]
                    pred_da = np.argmax(sample_das[r_id], axis=1)[0]
                    pred_tokens = [self.vocab[e] for e in pred_outs[b_id].tolist() if e != self.eos_id and e != 0]
                    pred_str = " ".join(pred_tokens).replace(" ' ", "'")
                    dest.write("Sample %d (%s) >> %s\n" % (r_id, self.da_vocab[pred_da], pred_str))
                    local_tokens.append(pred_tokens)

                max_bleu, avg_bleu = utils.get_bleu_stats(true_tokens, local_tokens)
                recall_bleus.append(max_bleu)
                prec_bleus.append(avg_bleu)
                # make a new line for better readability
                dest.write("\n")

        avg_recall_bleu = float(np.mean(recall_bleus))
        avg_prec_bleu = float(np.mean(prec_bleus))
        avg_f1 = 2*(avg_prec_bleu*avg_recall_bleu) / (avg_prec_bleu+avg_recall_bleu+10e-12)
        report = "Avg recall BLEU %f, avg precision BLEU %f and F1 %f (only 1 reference response. Not final result)" \
                 % (avg_recall_bleu, avg_prec_bleu, avg_f1)
        print(report)
        dest.write(report + "\n")
        print("Done testing")


