# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=0 # change to your gpu number

task=humaneval
length=256
block_length=32
initial_block_length=48
initial_steps=16
max_steps=32
max_block=96
min_block=12
steps=$((length / block_length))
model_path='GSAI-ML/LLaDA-8B-Instruct'

# cadllm (for the main results table)
accelerate launch \
  --num_processes 1 \
  eval_llada.py \
  --tasks ${task} \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args \
    "model_path=${model_path},\
    gen_length=${length},\
    initial_block_length=${initial_block_length},\
    use_cache=True,\
    initial_steps=${initial_steps},\
    max_steps=${max_steps},\
    max_block=${max_block},\
    min_block=${min_block},\
    confidence_method=softmax,\
    adaptive_blocks=True,\
    adaptive_steps=True,\
    adaptive_vocab_size=True,\
    adaptive_threshold=True,\
    show_speed=True" \
  --output_path evals_results/cadllm/humaneval-ns0-${length} \
  --log_samples

## NOTICE: use postprocess for humaneval
# python postprocess_code.py ./evals_results/<file_path>.jsonl
