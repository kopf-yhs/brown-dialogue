#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import os

import torch
from parlai.agents.hugging_face.dict import Gpt2DictionaryAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.core.torch_agent import TorchAgent, Batch, Output, DictionaryAgent
from parlai.core.brownian import BrownianBridgeLoss
from parlai.utils.io import PathManager
from parlai.utils.misc import warn_once
from parlai.utils.torch import IdentityLayer, padded_tensor

import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    GPT2Tokenizer,
    GPT2TimeLMHeadModel
)
from parlai.utils.torch import (
    neginf,
    IdentityLayer
)
from typing import Tuple
from math import ceil

from transformers import GPT2TimeModel, GPT2Model, AutoTokenizer


MAX_LENGTH = int(10000) 
HIDDEN_DIM = 128

def load_cl_model(filepath, latent_dim, base_model, use_section_ids,
                  token_size, pad_idx=0):
    model = GPT2OUEncoder(
         hidden_dim=HIDDEN_DIM,
         latent_dim=latent_dim,
         finetune_gpt2=False,
         padding_idx=pad_idx)
    if use_section_ids:
        model.model.resize_token_embeddings(token_size)

    state_dict = torch.load(filepath)
    new_dict = {}
    for k, v in state_dict['state_dict'].items():
        if any([i in k for i in ['model.model.g_ar', 'model.model.W_k']]):
            new_dict[k[6:]] = v
        elif any([i in k for i in ['model.g_ar', 'model.W_k', 'time_model']]):
            continue
        elif "model." in k:
            new_dict[k[6:]] = v
        else:
            new_dict[k] = v

    if any(['g_ar' in k for k in new_dict.keys()]):
        model.g_ar = torch.nn.GRU(input_size=latent_dim,
                           hidden_size=2400, # default number in infoNCE for langauge
                           num_layers=3,
                           batch_first=True
                           )
        model.W_k = torch.nn.Linear(2400, latent_dim)
    elif any(['time_model' in k for k in state_dict['state_dict'].keys()]):
        model.fc_mu = torch.nn.Linear(latent_dim, latent_dim)
        model.fc_var = torch.nn.Linear(latent_dim, latent_dim)

    model.load_state_dict(new_dict)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

def get_checkpoint(
        filepath, 
        latent_dim, 
        base_model="gpt2",
        sec_id=False, 
        token_size=None,
        pad_idx=0,
    ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_cl_model(filepath,
                          latent_dim,
                          base_model,
                          use_section_ids=sec_id,
                          token_size=token_size,
                          pad_idx=pad_idx
                          )
    model.to(device)
    model = model.eval()
    return model

def assert_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks))

    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]

    # Duplicate check
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    # Missing blocks
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]

    assert len(duplicate_blocks) == 0, (
        "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device. These "
        "attention blocks were specified more than once: " + str(duplicate_blocks)
    )
    assert len(missing_blocks) == 0, (
        "There are attention blocks for this model that are not specified in the device_map. Add these attention "
        "blocks to a device on the device_map: " + str(missing_blocks)
    )
    assert (
        len(extra_blocks) == 0
    ), "The device_map contains more attention blocks than this model has. Remove these from the device_map:" + str(
        extra_blocks
    )

def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(devices, layers_list))

def get_special_tokens(dataset_name, tokenizer, add_tokens=True):
    SECTION_IDS = []
    if "wikisection" == dataset_name:
        SECTION_IDS = ['[ ABSTRACT ]', '[ HISTORY ]', '[ GEOGRAPHY ]', '[ DEMOGRAPHICS ]']
    if 'recipe' in dataset_name:
        SECTION_IDS = [
            '[ TITLE ]',
            '[ INGREDIENTS ]',
            '[ DIRECTIONS ]'
        ]
    if 'tm2' in dataset_name or 'tickettalk' in dataset_name:
        SECTION_IDS = [
            '[ USER ]',
            '[ ASSISTANT ]',
        ]
    if 'wikihow' in dataset_name:
        SECTION_IDS = [
            '[ TITLE ]',
            '[ METHOD ]',
            '[ STEP ]'
        ]
    SECTION_IDS += [' . ']
    if add_tokens:
        # NOTE loading previous tokenizer sometimes already includes the new tokens
        eos = tokenizer(' . ')['input_ids']
        print("Old tokenizer size: ", len(tokenizer))
        if len(eos) == 1 and eos[0] == 50256 + len(SECTION_IDS):
            print("Not adding because it's already contained")
            pass # don't add cause it's already contained
        else:
            print("Adding tokens, ", SECTION_IDS)
            tokenizer.add_tokens(SECTION_IDS)
        print("New tokenizer size: ", len(tokenizer))
    SPECIAL_TOKENS = [_[0] for _ in tokenizer(SECTION_IDS)['input_ids']]
    return SECTION_IDS, SPECIAL_TOKENS, tokenizer

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.zeros_(m.bias)
        m.bias.requires_grad = False

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def simulate_brownian_bridge(B_0, B_T, num_samples, sentence_lengths, dt=0.05, mu=0.0, sigma=1.0):
    """Run bridge forward pinned at B_0 and B_T"""
    if isinstance(B_0, torch.Tensor):
        B_0 = B_0.cpu().detach().numpy()
    if isinstance(B_T, torch.Tensor):
        B_T = B_T.cpu().detach().numpy()

    bridge = [B_0]
    x_t = np.copy(B_0)
    for step in range(num_samples - 2): # number of sentences
        dim = B_0.shape[-1]
        noise = np.sqrt(dt)*sigma*np.random.normal(mu, sigma, dim)
        t = step/num_samples
        x_tp1 = x_t * (1- dt/(1-t)) + (dt/(1-t))*B_T + noise
        length_idx = step % len(sentence_lengths)
        bridge += [x_tp1] * sentence_lengths[length_idx]
        x_t = x_tp1

    length_idx = step % len(sentence_lengths)
    bridge += [B_T] * sentence_lengths[length_idx]

    return bridge

def split_text(raw_text):
    split_pattern = ". "
    split_raw_text = [_ + split_pattern for _ in raw_text.split(split_pattern)]
    split_raw_text[-1] = split_raw_text[-1].rstrip(split_pattern)
    return split_raw_text

def get_density(dataset, lm, cl_model):
    """Estimate density of last latent"""
    first_latents = []
    last_latents = []
    length = len(dataset)
    for text_i in range(length):
        first_latents.append(dataset.cl_embeddings[text_i][0].detach().cpu().numpy())
        last_latents.append(dataset.cl_embeddings[text_i][-1].detach().cpu().numpy())
    first_latents = np.array(first_latents)
    last_latents = np.array(last_latents)
    return first_latents.mean(0), first_latents.std(0), last_latents.mean(0), last_latents.std(0)

############################################
## Modules
############################################

class GPT2OUEncoder(torch.nn.Module):
    '''
    GPT-2 Encoder for Brownian Bridge construction. 
    Passes GPT-2 hidden states through linear layers for Brownian Bridge latent representations.
    '''

    def __init__(self, hidden_dim, latent_dim, finetune_gpt2=False, padding_idx = None):
        super(GPT2OUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.finetune = finetune_gpt2
        self.padding_idx = padding_idx
        self._init_model()

    def _init_model(self):
        self.model = GPT2Model.from_pretrained('gpt2')
        self.model = self.model.eval()
        # turn off all the gradients
        for param in self.model.parameters():
            param.requires_grad = self.finetune
        self.mlp = torch.nn.Linear(self.model.wte.embedding_dim, self.hidden_dim)
        self.feature_extractor = self.create_feature_extractor() # data_dim -> hidden_dim
        self.log_q = self.create_log_q()
        self.C_eta = torch.nn.Linear(1, 1)

        ## NEW AUG 19, turn off bias training.
        self.mlp.apply(weights_init)
        self.feature_extractor.apply(weights_init)
        self.log_q.apply(weights_init)
        self.C_eta.apply(weights_init)

    def create_feature_extractor(self):
        return torch.nn.Sequential(*[
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.latent_dim),
                               ])

    def create_log_q(self):
        return torch.nn.Sequential(*[
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.Linear(self.latent_dim, 1),
                               ])

    def get_gpt2_embeddings(self, input_ids, attention_mask):
        gpt_emb = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        return gpt_emb

    def get_log_q(self, x):
        return self.log_q(x)

    def set_to_train(self):
        pass

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def projection(self, gpt_emb):
        z = self.mlp(gpt_emb) # 32, 100
        z = self.feature_extractor(z)
        return z

    def get_attn_mask(self, tokens_tensor):
        attn_mask = (tokens_tensor != self.padding_idx)
        return attn_mask

    def forward(self, input_ids):
        attn_mask = self.get_attn_mask(input_ids)
        gpt_emb = self.model(input_ids=input_ids, attention_mask=attn_mask)[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        gpt_emb = self.compute_masked_means(gpt_emb, attn_mask)
        # Albert lang embedding -> feature embedding space
        return self.projection(gpt_emb)

class GPT2TimeDecoder(torch.nn.Module):
    def __init__(self, opt, dict):
        super().__init__()
        self.transformer = self._init_from_pretrained(opt)
        
        '''
        if opt["add_special_tokens"]:
            size_before = self.transformer.wte.weight.size(0)
            self.transformer.resize_token_embeddings(len(dict.hf_tokenizer))
            with torch.no_grad():
                # first reduce the random jitter of the initialization
                self.transformer.wte.weight[size_before:] *= 0.1
                # next center it on the endoftext token
                self.transformer.wte.weight[
                    size_before:
                ] += self.transformer.wte.weight[size_before - 1].unsqueeze(0)
        '''

        self.add_start_token = opt["add_start_token"]
        self.START_IDX = dict.start_idx
        self.NULL_IDX = dict.null_idx
        self.END_IDX = dict.end_idx
        # use cuda
        self.use_cuda = not opt["no_cuda"] and torch.cuda.is_available()
    
    def _init_from_pretrained(self, opt):
        return GPT2TimeLMHeadModel.from_pretrained(opt['decoder_model_name'])

    def forward(self, input, encoder_state, cl_feats, incr_state=None):                
        attention_mask = None
        position_ids = None
        if incr_state is None:
            # first step
            if (
                not self.add_start_token
                and input.size(1) == 1
                and int(input[0][0]) == self.START_IDX
            ):
                # generating: ignore the start token
                # without deep copy, the padding_idx (-1) in encoder_state can be reset to 0 with clamp_ inplace operation
                model_input = encoder_state.clone()
            else:
                # forced decoding: concatenate the context
                # with the labels
                model_input = torch.cat([encoder_state, input], dim=-1)
            attention_mask = model_input != self.NULL_IDX
            position_ids = (
                attention_mask.cumsum(dim=-1, dtype=torch.int64) - 1
            ).clamp_(min=0)
        else:
            if not self.add_start_token:
                input = input[:, 1:]
            # generating with continuation
            # get the position ids
            position_ids = (encoder_state != self.NULL_IDX).sum(
                -1, True, dtype=torch.int64
            ) - 1
            delta = ((input != self.NULL_IDX)).sum(-1, True, dtype=torch.int64)
            position_ids += delta
            # generation: get the last token input
            model_input = input[:, -1:]
            attention_mask = torch.cat([encoder_state, input], dim=-1) != self.NULL_IDX

        model_input = model_input.clamp_(min=0)
        transformer_outputs = self.transformer(
            input_ids=model_input,
            cl_feats=cl_feats,
            past_key_values=incr_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        new_incr_state = transformer_outputs[1]

        if incr_state is None:
            # pull out only the hidden states for the label tokens
            output = hidden_states[:, -input.size(1) - 1 + int(self.add_start_token) :]
            # hack: we need the last state of the encoder-side to be the first
            # element of the decoder-side
            lengths = (input != self.NULL_IDX).sum(dim=-1)
            for i in range(input.size(0)):
                output[i, input.size(1) - lengths[i]] = output[i, 0]

        else:
            # generation, we're only doing one token at a time. no need to
            # shove things back in
            output = hidden_states

        return output, new_incr_state

class BrownianGPT2Model(TorchGeneratorModel):
    """
    Implementation of Language Modeling via Stochasitc Processes for Dialogue Generation
    """

    def __init__(self, opt, dict):
        self.add_start_token = opt["add_start_token"]
        super().__init__(*self._get_special_tokens(opt, dict))

        # init the model
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        SECTION_IDS, SPECIAL_TOKENS, self.tokenizer = get_special_tokens(
            dataset_name=opt['dataset_name'], tokenizer=self.tokenizer)
        token_size = len(self.tokenizer)

        self.encoder = IdentityLayer()
        self.cl_encoder = self._get_encoder(opt, token_size=token_size)
        self.decoder = self._get_decoder(opt, dict)
        self.config = self.decoder.transformer.config
        
        # add start token
        # temporary fix to manually retrieve pre-computed Gaussian estimate
        # TO-DO : decide how to get Gaussian estimate for last sentence in an interactive mode.
        import pickle
        with open(opt['gaussian_path'], 'rb') as f:
            self.first_sent_mu, self.first_sent_dev, self.last_sent_mu, self.last_sent_dev = pickle.load(f)

    def _get_encoder(self, opt, token_size):
        model_path = opt['encoder_model_name']
        latent_dim = opt['encoder_latent_dim']
        return get_checkpoint(
            model_path, 
            latent_dim,
            sec_id=True,
            token_size=token_size,
            pad_idx=self.tokenizer.pad_token_id
        )

    def _get_decoder(self, opt, dict):
        assert 'decoder_model_name' in opt.keys()
        return GPT2TimeDecoder(opt, dict)

    def _tie_weights(self, output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.weight

    def _get_special_tokens(self, opt, dict):
        return dict.null_idx, dict.start_idx, dict.end_idx

    def reorder_encoder_states(self, encoder_states, indices):
        enc = torch.index_select(encoder_states, 0, indices)
        return enc

    #def output(self, tensor):
    #    """
    #    Compute output logits.
    #    """
    #    return self.lm_head(tensor)

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        new_incr_state = []
        for layer_past in incremental_state:
            if torch.is_tensor(layer_past):
                new_incr_state.append(torch.index_select(layer_past, 1, inds))
            else:
                # newer versions of HF split up the intermediate outputs
                assert isinstance(layer_past, tuple)
                layer_past = torch.stack(layer_past, dim=0)
                new_incr_state.append(torch.index_select(layer_past, 1, inds))

        return tuple(new_incr_state)

    def decode_forced(self, encoder_states, ys):
        """
        Override to get rid of start token input.
        """
        #if self.add_start_token:
        #    return super().decode_forced(encoder_states, ys)
        
        # Estimate dnesity for last sentence
        #first_latent_mu, first_latent_std, last_latent_mu, last_latent_std = get_density(dataset=train_dataset, lm=model, cl_model=CL_MODEL)
        B_0 = np.random.normal(loc=self.first_sent_mu, scale=self.first_sent_dev)
        B_T = np.random.normal(loc=self.last_sent_mu, scale=self.last_sent_dev)
        bridge_feats = simulate_brownian_bridge(
            B_0=B_0, B_T=B_T, num_samples=10,
            sentence_lengths=10
        )
        cl_feats = bridge_feats[0]

        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        latent, _ = self.decoder(
            inputs, 
            encoder_states,
            cl_feats = cl_feats,
            seq_cl_feats = bridge_feats,
            encoder_hidden_states=encoder_states,
        )
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

############################################
## Agent
############################################

class BrownianAgent(TorchGeneratorAgent):
    """
    Hugging Face GPT2 Agent.

    GPT2 is a multi-layer decoder-only Transformer.
    The decoder is initialized with pretrained weights from Hugging Face.
    Read more about this model here
    <https://huggingface.co/transformers/model_doc/gpt2.html>.

    GPT2 comes in five sizes: distilgpt2, small, medium, large, XL. Use the
    flag `--gpt2-size` to choose the size.

    If you are finetuning the Gpt2 agent as a dialogue agent, be sure
    to run `--add-special-tokens True`. To examine the performance of the
    agent out of the box, run with `--add-special-tokens False`, and make
    sure that the batch size is 1.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group("Gpt2 Args")
        agent.add_argument(
            "--model-name", type=str, default=None, help="Any GPT-2 model names."
        )
        agent.add_argument(
            "--gpt2-size",
            type=str,
            default="small",
            choices=["small", "medium", "large", "xl", "distilgpt2"],
            help="Which size model to initialize.",
        )
        agent.add_argument(
            "--add-special-tokens",
            type="bool",
            default=True,
            help="Add special tokens (like PAD, etc.). If False, "
            "Can only use with batch size 1.",
        )
        agent.add_argument(
            "--add-start-token",
            type="bool",
            default=True,
        )
        agent.add_argument(
            "--encoder-model-name",
            type=str,
            required=True,
        )
        agent.add_argument(
            "--encoder-latent-dim",
            type=int,
            default=8,
        )
        agent.add_argument(
            "--decoder-model-name",
            type=str,
            required=True,
        )
        agent.add_argument(
            '--dataset_name',
            type=str,
            required=True,
        )
        agent.add_argument(
            "--gaussian-path",
            type=str,
        )
        parser.set_defaults(
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        warn_once("WARNING: this model is in beta and the API is subject to change.")
        return agent

    def __init__(self, opt, shared=None):
        if not opt["add_special_tokens"] and opt.get('batchsize', 1) > 1:
            # *** STOP ***
            # You may be a future researcher who has stumbled upon this odd
            # restriction, and is tempted to comment this out. After all, the
            # code still runs when it's uncommented, why shouldn't you?
            # You should know this has serious implications, as gpt2 doesn't have
            # padding tokens. This is incompatible with ParlAI's batching,
            # which puts conversations of different length in the same
            # batch. Without a padding token, nonsense will be inserted into
            # the context, and the generations & PPL you get will be wrong.
            raise RuntimeError(
                "If using batchsize > 1, --add-special-tokens must be True."
            )
        if not opt["add_special_tokens"] and opt["add_start_token"]:
            raise RuntimeError(
                "--add-start-token true requires --add-special-tokens true"
            )
        super().__init__(opt, shared)
        if hasattr(self.model, "module"):
            self.START_IDX = self.model.module.START_IDX
            self.END_IDX = self.model.module.END_IDX
            self.NULL_IDX = self.model.module.NULL_IDX
        else:
            self.START_IDX = self.model.START_IDX
            self.END_IDX = self.model.END_IDX
            self.NULL_IDX = self.model.NULL_IDX

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overridden if a more complex dictionary is required.
        """
        return Gpt2DictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return BrownianGPT2Model(self.opt, self.dict)

    def _encoder_input(self, batch):
        return (batch.text_vec,)

    def _pad_tensor(self, items, is_label=False):
        """
        Override to always set fp16friendly to False and left_pad to True.
        """
        return padded_tensor(
            items, pad_idx=self.NULL_IDX, left_padded=True, fp16friendly=False
        )

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ):
        
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        encoder_states = model.encoder(*self._encoder_input(batch))
        # get Brownian embedding
        cl_encoder = self.model.cl_encoder
        cl_feats = cl_encoder(*self._encoder_input(batch))
        
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = batch.batchsize
        if batch.text_vec is not None:
            batchsize = batch.batchsize
            batch_context_list = self._get_batch_context(batch)
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                .set_batch_context(
                    batch_context_list,
                    batch_idx,
                    self.opt.get('gpu_beam_blocking', False),
                )
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                for _ in range(bsz)
            ]
        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = model.decoder(decoder_input, encoder_states, cl_feats, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = self._generation_activation(score)  # type: ignore
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts]
                prefix_mask = torch.ones_like(score, dtype=torch.bool)
                prefix_mask[
                    :, :, prefix_toks
                ] = False  # everything except prefix toks should be neginf
                score[prefix_mask] = neginf(score.dtype)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i], _ts)
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams

    ### TO-DO : Enable training on ParlAI framework

    #def compute_loss(self, batch, batch_idx, return_output=False):
    #    torch.cuda.empty_cache()
    #    obs_0 = batch['y_0']
    #    obs_t = batch['y_t']
    #    obs_T = batch['y_T']
    #    t_s = batch['t_'].float()
    #    ts = batch['t'].float()
    #    Ts = batch['T'].float()
    #    feats_0 = self.get_feats(obs_0)
    #    feats_t = self.get_feats(obs_t)
    #    feats_T = self.get_feats(obs_T)
    #    log_q_y_tp1 = self.model.get_log_q(feats_t)
    #    loss_fn = brownian_bridge.BrownianBridgeLoss(
    #        z_0=feats_0,
    #        z_t=feats_t,
    #        z_T=feats_T,
    #        t_=t_s,
    #        t=ts,
    #        T=Ts,
    #        alpha=0,
    #        var=0,
    #        log_q_y_T=log_q_y_tp1,
    #        loss_type=self.config.loss_params.name,
    #        eps=self.config.model_params.eps,
    #        max_seq_len=batch['total_t'].float(),
    #    )
    #    loss = loss_fn.get_loss()
    #    return loss

    #def train_step(self, batch, batch_idx):
    #    loss = self.compute_loss(batch, batch_idx)
    #    return loss

    #def test_step(self, batch, i):
    #    loss = self.get_losses_for_batch(batch=batch, batch_idx=i)
    #    wandb.log({'test_loss': loss.cpu().detach().numpy(),
    #               'epoch': self.trainer.current_epoch})
    #    self.log('test_loss', float(loss.cpu().detach().numpy()), prog_bar=True, on_step=True)
    #    return loss

    #def forward(self, input_ids, attention_mask):
    #    feats = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
    #    return feats

    #def get_feats(self, obs):
    #    input_ids_i, attention_mask_i = self.train_dataset.tokenize_caption(
    #        obs, device=self.device)
    #    input_ids_i = input_ids_i[:, :self.train_dataset.max_length]
    #    attention_mask_i = attention_mask_i[:, :self.train_dataset.max_length]
    #    feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
    #    return feats_i

    #def compute_loss(self, batch, return_output=False):
    #    pass

    def load_state_dict(self, state_dict):
        # 2020-11-10: some very old transformer model points (pre v3.0.1) are
        # missing a field called transformer.h.0.attn.masked_bias. This hacks
        # around that. See
        # https://github.com/huggingface/transformers/issues/4309.
        current_sd = self.model.state_dict()
        missing = set(current_sd.keys()) - set(state_dict.keys())
        for m in missing:
            if 'masked_bias' in m:
                state_dict[m] = current_sd[m]
        return super().load_state_dict(state_dict)
