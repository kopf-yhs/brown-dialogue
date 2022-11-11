#!/bin/bash

parlai display_data --task taskmaster3 -n 10

parlai interactive --model brownian/brownian \
    --add_start_token True \
    --add_special_tokens True \
    --encoder-model-name /home/yongho/nas/BrownianDialogue/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/outputs/2022-11-10/20-44-10/brownian_bridge8_tickettalk/lightning_logs/version_0/checkpoints/epoch=4-step=3904.ckpt \
    --decoder-model-name /home/yongho/nas/BrownianDialogue/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/transformers/examples/pytorch/language-modeling/LM_tickettalk_8 \
    --gaussian_path /home/yongho/nas/BrownianDialogue/language_modeling_via_stochastic_processes/LM_tickettalk_8_density.pkl \
