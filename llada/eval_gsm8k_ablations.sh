# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=0 # change to your gpu number

task=gsm8k
length=256
block_length=32
initial_block_length=24
base_steps=24
max_steps=90
max_block=64
min_block=4
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'

# the following ablations turn one adaptive policy on at a time

# ablations - adaptive threshold
accelerate launch \
  --num_processes 1 \
  eval_llada.py \
  --tasks ${task} \
  --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args \
    "model_path=${model_path},\
    gen_length=${length},\
    initial_block_length=${initial_block_length},\
    use_cache=True,\
    base_steps=${base_steps},\
    max_steps=${max_steps},\
    max_block=${max_block},\
    min_block=${min_block},\
    confidence_method=softmax,\
    show_speed=True,\
    adaptive_blocks=False,\
    adaptive_vocab_size=False,\
    adaptive_steps=False,\
    adaptive_threshold=True" \
  --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-adaptive_threshold_on \
  --log_samples

# ablations - adaptive blocks
accelerate launch \
  --num_processes 1 \
  eval_llada.py \
  --tasks ${task} \
  --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args \
    "model_path=${model_path},\
    gen_length=${length},\
    initial_block_length=${initial_block_length},\
    use_cache=True,\
    base_steps=${base_steps},\
    max_steps=${max_steps},\
    max_block=${max_block},\
    min_block=${min_block},\
    confidence_method=softmax,\
    show_speed=True,\
    adaptive_blocks=True,\
    adaptive_vocab_size=False,\
    adaptive_steps=False,\
    adaptive_threshold=False" \
  --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-adaptive_blocks_on \
  --log_samples

# ablations - adaptive steps
accelerate launch \
  --num_processes 1 \
  eval_llada.py \
  --tasks ${task} \
  --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args \
    "model_path=${model_path},\
    gen_length=${length},\
    initial_block_length=${initial_block_length},\
    use_cache=True,\
    base_steps=${base_steps},\
    max_steps=${max_steps},\
    max_block=${max_block},\
    min_block=${min_block},\
    confidence_method=softmax,\
    show_speed=True,\
    adaptive_blocks=False,\
    adaptive_vocab_size=False,\
    adaptive_steps=True,\
    adaptive_threshold=False" \
  --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-adaptive_steps_on \
  --log_samples

# ablations - adaptive vocab size
accelerate launch \
  --num_processes 1 \
  eval_llada.py \
  --tasks ${task} \
  --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args \
    "model_path=${model_path},\
    gen_length=${length},\
    initial_block_length=${initial_block_length},\
    use_cache=True,\
    base_steps=${base_steps},\
    max_steps=${max_steps},\
    max_block=${max_block},\
    min_block=${min_block},\
    confidence_method=softmax,\
    show_speed=True,\
    adaptive_blocks=False,\
    adaptive_vocab_size=True,\
    adaptive_steps=False,\
    adaptive_threshold=False" \
  --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-adaptive_vocab_size_on \
  --log_samples