#!/bin/bash
nseed=1

domains=("tickettalk")
path2repo="/convei_nas/yongho/BrownianDialogue/language_modeling_via_stochastic_processes"
encoder_filepath="/convei_nas/yongho/BrownianDialogue/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/models/tickettalk/tc8/epoch=4-step=3904.ckpt"
latent_dims=(8)

for domain in ${domains[@]}; do
    for seed in $(seq 1 1 $nseed); do
        for latent_dim in ${latent_dims[@]}; do
            python store_gaussian.py --model_type=gpt2 --model_name_or_path=${path2repo}/language_modeling_via_stochastic_processes/transformers/examples/pytorch/language-modeling/LM_${domain}_${latent_dim}/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=${domain} --encoder_filepath=${path2repo}/language_modeling_via_stochastic_processes/models/${domain}/tc${latent_dim}/epoch=99-step=75299.ckpt --latent_dim=${latent_dim} --project=LM_${domain} --no_eos --label=LM_${domain}_${latent_dim} --seed=${seed}
        done
    done
done
