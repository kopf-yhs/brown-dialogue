Work in Progress. Credits to [ParlAI](http://parl.ai) and [rosewang2008](https://github.com/rosewang2008/language_modeling_via_stochastic_processes/) for open-sourcing their projects.

What's done:
- Load encoder & decoder checkpoints fine-tuned using [rosewang2008](https://github.com/rosewang2008/language_modeling_via_stochastic_processes/) and run inference (interactive) on ParlAI
- Template (see ```test.sh```)
```bash
parlai interactive --model brownian/brownian \
    --add_start_token True \
    --add_special_tokens True \
    --dataset_name tm2 ("Ticketmaster2") \
    --encoder_model_name "Path to BrownianBridge encoder" \
    --decoder_model_name "Path to BrownianBridge decoder" \
    --gaussian_path "Path to Gaussian of BrownianBridge embeddings for start and end sentences in the target dataset" \

```

TO-DO:
- Enable encoder & decoder training on ParlAI framework