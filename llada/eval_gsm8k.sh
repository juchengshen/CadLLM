# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=7 # change to your gpu number

task=gsm8k
length=256 # default is 256, change to 512 or more when needed
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'

# cadllm (for the main results table)
accelerate launch --num_processes 1 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},initial_block_length=24,use_cache=True,initial_steps=24,max_steps=90,max_block=64,min_block=4,confidence_method=softmax,adaptive_blocks=True,adaptive_steps=True,adaptive_vocab_size=True,adaptive_threshold=True,show_speed=True

# ------------------------

# current submission's ablations (ON + OFF + three out of four adaptive policies enabled at a time), we will change to one adaptive policy enabled at a time for rebuttal

# ablations - ON (adaptive blocks, adaptive steps, adaptive vocab size, adaptive threshold)
# accelerate launch --num_processes 1 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},initial_block_length=24,use_cache=True,initial_steps=24,max_steps=90,max_block=64,min_block=4,confidence_method=softmax,show_speed=True,adaptive_blocks=True,adaptive_vocab_size=True,adaptive_steps=True,adaptive_threshold=True \
# --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-ablation-on --log_samples

# ablations - adaptive blocks, adaptive steps, no adaptive vocab size
# accelerate launch --num_processes 1 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},initial_block_length=24,use_cache=True,initial_steps=24,max_steps=90,max_block=64,min_block=4,confidence_method=softmax,show_speed=True,adaptive_blocks=True,adaptive_vocab_size=False,adaptive_steps=True \
# --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-ablation-no-vocab_size --log_samples

# ablations - adaptive blocks, adaptive vocab size, no adaptive steps
# accelerate launch --num_processes 1 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},initial_block_length=24,use_cache=True,initial_steps=24,max_steps=90,max_block=64,min_block=4,confidence_method=softmax,show_speed=True,adaptive_blocks=True,adaptive_vocab_size=True,adaptive_steps=False \
# --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-ablation-no-steps --log_samples

# ablations - adaptive steps, adaptive vocab size, no adaptive blocks
# accelerate launch --num_processes 1 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},initial_block_length=24,use_cache=True,initial_steps=24,max_steps=90,max_block=64,min_block=4,confidence_method=softmax,show_speed=True,adaptive_blocks=False,adaptive_vocab_size=True,adaptive_steps=True \
# --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-ablation-no-blocks --log_samples

# ablations - adaptive steps, adaptive vocab size, no adaptive threshold
# accelerate launch --num_processes 1 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},initial_block_length=24,use_cache=True,initial_steps=24,max_steps=90,max_block=64,min_block=4,confidence_method=softmax,show_speed=True,adaptive_blocks=False,adaptive_vocab_size=True,adaptive_steps=True,adaptive_threshold=False \
# --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-ablation-no-threshold --log_samples

# ablations - OFF (no adaptive steps, no adaptive vocab size, no adaptive blocks, no adaptive threshold)
# accelerate launch --num_processes 1 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},initial_block_length=24,use_cache=True,initial_steps=24,max_steps=90,max_block=64,min_block=4,confidence_method=softmax,show_speed=True,adaptive_blocks=False,adaptive_vocab_size=False,adaptive_steps=False,adaptive_threshold=False \
# --output_path evals_results/cadllm/gsm8k-ns${num_fewshot}-${length}-ablation-off --log_samples