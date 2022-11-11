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
from parlai.utils.io import PathManager
from parlai.utils.misc import warn_once
from parlai.utils.brown import BrownianBridgeLoss, BrownianLoss
from parlai.utils.torch import IdentityLayer, padded_tensor

import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    GPT2Tokenizer
)
from typing import Tuple
from math import ceil

from transformers import GPT2TimeModel, GPT2Model


MAX_LENGTH = int(10000) 
HIDDEN_DIM = 128

def load_cl_model(filepath, latent_dim, base_model, use_section_ids,
                  token_size):
    model = GPT2OUEncoder(
         hidden_dim=HIDDEN_DIM,
         latent_dim=latent_dim,
         finetune_gpt2=False)
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

    print(new_dict)
    model.load_state_dict(new_dict)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

def get_checkpoint(filepath, latent_dim, base_model="gpt2",
                   sec_id=False, token_size=None,
                ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_cl_model(filepath,
                          latent_dim,
                          base_model,
                          use_section_ids=sec_id,
                          token_size=token_size
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

    def __init__(self, hidden_dim, latent_dim, finetune_gpt2=False):
        super(GPT2OUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.finetune = finetune_gpt2
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

    def forward(self, input_ids, attention_mask):
        gpt_emb = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        # Albert lang embedding -> feature embedding space
        return self.projection(gpt_emb)


class GPT2TimeDecoder(torch.nn.Module):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, opt, dict):
        self.transformer = self._init_from_pretrained(opt)
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

        self.add_start_token = opt["add_start_token"]
        self.START_IDX = dict.start_idx
        self.NULL_IDX = dict.null_idx
        self.END_IDX = dict.end_idx
        # use cuda
        self.use_cuda = not opt["no_cuda"] and torch.cuda.is_available()

    def _init_from_pretrained(self, opt):
        # load model
        if opt.get("decoder_model_name"):
            fle_key = opt["decoder_model_name"]
        else:
            model_sz = opt["gpt2_size"]
            if model_sz == "small":
                model_key = "gpt2"
            elif model_sz == "distilgpt2":
                model_key = "distilgpt2"
            else:
                model_key = f"gpt2-{model_sz}"

            # check if datapath has the files that hugging face code looks for
            hf_dir = os.path.join(opt["datapath"], "hf", model_key)
            if all(
                PathManager.exists(os.path.join(hf_dir, file_name))
                for file_name in ["pytorch_model.bin", "config.json"]
            ):
                fle_key = PathManager.get_local_path(hf_dir, recursive=True)
            else:
                fle_key = model_key
        return GPT2TimeModel.from_pretrained(fle_key)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        result = {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        if 'section_ids' in kwargs:
            result['section_ids']= kwargs['section_ids']
        if 'raw_text' in kwargs:
            result['raw_text']= kwargs['raw_text']
        if 'cl_feats' in kwargs:
            result['cl_feats']= kwargs['cl_feats']
        if 'seq_cl_feats' in kwargs:
            result['seq_cl_feats']= kwargs['seq_cl_feats']
        if 'seq_section_ids' in kwargs:
            result['seq_section_ids']= kwargs['seq_section_ids']

        return result

    def forward(
        self,
        input_ids=None,
        raw_text=None,
        seq_cl_feats=None,
        seq_section_ids=None,
        cl_feats=None,
        section_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            raw_text=raw_text,
            cl_feats=cl_feats,
            seq_cl_feats=seq_cl_feats,
            seq_section_ids=seq_section_ids,
            section_ids=section_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

class BrownianGPT2Model(TorchGeneratorModel):
    """
    Implementation of Language Modeling via Stochasitc Processes for Dialogue Generation
    """

    def __init__(self, opt, dict):
        print(opt.items())
        self.add_start_token = opt["add_start_token"]
        super().__init__(*self._get_special_tokens(opt, dict))

        # init the model
        self.encoder = self._get_encoder(opt, dict)
        self.decoder = self._get_decoder(opt, dict)
        self.config = self.decoder.transformer.config
        self.lm_head = torch.nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False
        )
        self._tie_weights(self.lm_head, self.decoder.transformer.wte)
        # add start token

        # temporary fix to manually retrieve pre-computed Gaussian estimate
        # TO-DO : decide how to get Gaussian estimate for last sentence in an interactive mode.
        import pickle
        with open(opt['gaussian_path'], 'rb') as f:
            self.first_sent_mu, self.first_sent_dev, self.last_sent_mu, self.last_sent_dev = pickle.load(f)

    def _get_encoder(self, opt, dict):
        model_path = opt['encoder_model_name']
        latent_dim = opt['encoder_latent_dim']
        return get_checkpoint(model_path, latent_dim)

    def _get_decoder(self, opt, dict):
        return GPT2TimeDecoder(opt, dict)

    def _tie_weights(self, output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.weight

    def _get_special_tokens(self, opt, dict):
        return dict.null_idx, dict.start_idx, dict.end_idx

    def reorder_encoder_states(self, encoder_states, indices):
        enc = torch.index_select(encoder_states, 0, indices)
        return enc

    def output(self, tensor):
        """
        Compute output logits.
        """
        return self.lm_head(tensor)

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
