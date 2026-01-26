# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=0 # change to your gpu number

task=mbpp
length=256
block_length=32
initial_block_length=24
base_steps=24
max_steps=90
max_block=64
min_block=4
steps=$((length / block_length))
num_fewshot=3
model_path='GSAI-ML/LLaDA-8B-Instruct'

# cadllm (for the main results table)
accelerate launch \
  --num_processes 1 \
  eval_llada.py \
  --tasks ${task} \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --num_fewshot ${num_fewshot} \
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
    adaptive_blocks=True,\
    adaptive_steps=True,\
    adaptive_vocab_size=True,\
    adaptive_threshold=True,\
    show_speed=True" \
  --output_path evals_results/cadllm/mbpp-ns${num_fewshot}-${length} \
  --log_samples