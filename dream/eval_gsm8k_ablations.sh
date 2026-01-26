# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=0 # change to desired gpu num

task=gsm8k
length=256
initial_block_length=16
initial_steps=24
max_steps=70
max_block=48
min_block=8
block_length=32
num_fewshot=5
steps=$((length / block_length))
model="Dream-org/Dream-v0-Base-7B"

# the following ablations turn one adaptive policy on at a time

# ablations - adaptive blocks)
accelerate launch \
  --num_processes=1 \
  eval.py \
  --model dream \
  --model_args \
    "pretrained=${model},\
    max_new_tokens=${length},\
    diffusion_steps=${length},\
    add_bos_token=True,\
    initial_block_length=${initial_block_length},\
    use_cache=True,\
    dual_cache=True,\
    initial_steps=${initial_steps},\
    max_steps=${max_steps},\
    max_block=${max_block},\
    min_block=${min_block},\
    confidence_method=softmax,\
    adaptive_blocks=True,\
    adaptive_steps=False,\
    adaptive_vocab_size=False,\
    adaptive_threshold=False" \
  --tasks ${task} \
  --num_fewshot ${num_fewshot} \
  --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-adaptive_blocks_on \
  --log_samples \
  --batch_size 1

# ablations - adaptive steps
accelerate launch \
  --num_processes=1 \
  eval.py \
  --model dream \
  --model_args \
    "pretrained=${model},\
    max_new_tokens=${length},\
    diffusion_steps=${length},\
    add_bos_token=True,\
    initial_block_length=${initial_block_length},\
    use_cache=True,\
    dual_cache=True,\
    initial_steps=${initial_steps},\
    max_steps=${max_steps},\
    max_block=${max_block},\
    min_block=${min_block},\
    confidence_method=softmax,\
    adaptive_blocks=False,\
    adaptive_steps=True,\
    adaptive_vocab_size=False,\
    adaptive_threshold=False" \
  --tasks ${task} \
  --num_fewshot ${num_fewshot} \
  --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-adaptive_steps_on \
  --log_samples \
  --batch_size 1

# ablations - adaptive vocab size
accelerate launch \
  --num_processes=1 \
  eval.py \
  --model dream \
  --model_args \
    "pretrained=${model},\
    max_new_tokens=${length},\
    diffusion_steps=${length},\
    add_bos_token=True,\
    initial_block_length=${initial_block_length},\
    use_cache=True,\
    dual_cache=True,\
    initial_steps=${initial_steps},\
    max_steps=${max_steps},\
    max_block=${max_block},\
    min_block=${min_block},\
    confidence_method=softmax,\
    adaptive_blocks=False,\
    adaptive_steps=False,\
    adaptive_vocab_size=True,\
    adaptive_threshold=False" \
  --tasks ${task} \
  --num_fewshot ${num_fewshot} \
  --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-adaptive_vocab_size_on \
  --log_samples \
  --batch_size 1

# ablations - adaptive threshold
accelerate launch \
  --num_processes=1 \
  eval.py \
  --model dream \
  --model_args \
    "pretrained=${model},\
    max_new_tokens=${length},\
    diffusion_steps=${length},\
    add_bos_token=True,\
    initial_block_length=${initial_block_length},\
    use_cache=True,\
    dual_cache=True,\
    initial_steps=${initial_steps},\
    max_steps=${max_steps},\
    max_block=${max_block},\
    min_block=${min_block},\
    confidence_method=softmax,\
    adaptive_blocks=False,\
    adaptive_steps=False,\
    adaptive_vocab_size=False,\
    adaptive_threshold=True" \
  --tasks ${task} \
  --num_fewshot ${num_fewshot} \
  --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-adaptive_threshold_on \
  --log_samples \
  --batch_size 1

# ablations - off
accelerate launch \
  --num_processes=1 \
  eval.py \
  --model dream \
  --model_args \
    "pretrained=${model},\
    max_new_tokens=${length},\
    diffusion_steps=${length},\
    add_bos_token=True,\
    initial_block_length=${initial_block_length},\
    use_cache=True,\
    dual_cache=True,\
    initial_steps=${initial_steps},\
    max_steps=${max_steps},\
    max_block=${max_block},\
    min_block=${min_block},\
    confidence_method=softmax,\
    adaptive_blocks=False,\
    adaptive_steps=False,\
    adaptive_vocab_size=False,\
    adaptive_threshold=False" \
  --tasks ${task} \
  --num_fewshot ${num_fewshot} \
  --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-off \
  --log_samples \
  --batch_size 1
