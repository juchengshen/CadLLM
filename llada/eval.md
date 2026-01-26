# LLaDA Evaluation

## Default hyperparameters

```bash
task=gsm8k
length=256 # or 512
block_length=32
initial_block_length=24
initial_steps=24
base_steps=24
max_steps=90
max_block=64
min_block=4
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'

use_cache=True
confidence_method=softmax
adaptive_blocks=True
adaptive_steps=True
adaptive_vocab_size=True
adaptive_threshold=True
show_speed=True
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