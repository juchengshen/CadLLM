# Dream Evaluation

## Default parameters

```bash
task=gsm8k
length=256 # or 512
initial_block_length=16
initial_steps=24
max_steps=70
max_block=48
min_block=8
block_length=32
num_fewshot=5
steps=$((length / block_length))
model="Dream-org/Dream-v0-Base-7B"

pretrained=${model}
max_new_tokens=${length}
diffusion_steps=${length}
add_bos_token=true
use_cache=true
dual_cache=true
confidence_method=softmax
adaptive_blocks=true
adaptive_steps=true
adaptive_vocab_size=true
adaptive_threshold=true
```

## Main scripts

```bash
bash eval_gsm8k.sh
bash eval_humaneval.sh
# After running humaneval, make sure you run postprocess to obtain final accuracy
python postprocess_code.py <path to .jsonl file under output_path>
bash eval_math.sh
bash eval_mbpp.sh
```

## Ablations

```bash
bash eval_gsm8k_ablations.sh
```