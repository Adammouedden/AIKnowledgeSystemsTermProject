#!/bin/bash
# One experiment arm on verl-agent, single A100 40 GB.
# Usage: run_arm.sh <name> <method> <rel> [extra hydra overrides...]
#   method: grpo | gaca      rel: none | msize
# Everything except algorithm.gaca.{method,rel} is shared by all arms and
# follows verl-agent's examples/gigpo_trainer/run_alfworld.sh (the paper's
# "public GiGPO recipe"); deviations forced by one 40 GB GPU are marked.
set -euo pipefail
unset GACA_TIME_BUDGET_H   # iterations are fixed (STEPS); the wall-clock cap is retired
NAME=$1; METHOD=$2; REL=$3; shift 3
ROOT=/workspace/AIKnowledgeSystemsTermProject
VA=/workspace/verl-agent
export HF_HOME=$ROOT/models ALFWORLD_DATA=$ROOT/data/alfworld
export TOKENIZERS_PARALLELISM=false  # recipe forces VLLM_ATTENTION_BACKEND=XFORMERS; its wheel does not load here, vLLM default (FlashAttention) is used
# The container caps the process tree at 2816 threads: keep per-process pools small.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAY_num_server_call_thread=1 ALF_ENVS_PER_ACTOR=8
OUT=$ROOT/src/runs/verl/$NAME; mkdir -p $OUT
cd $VA
. .venv/bin/activate

TRAIN_BS=${TRAIN_BS:-16}     # tasks per iteration (paper: 16)
VAL_BS=${VAL_BS:-128}        # held-out episodes (paper: 128)
STEPS=${STEPS:-34}           # iterations, fixed for every arm (paper: 150 epochs)
TEST_FREQ=${TEST_FREQ:-5}

python3 -m examples.data_preprocess.prepare --mode text --train_data_size $TRAIN_BS --val_data_size $VAL_BS

export GACA_VAL_DUMP=$OUT/val_episodes.jsonl   # per-episode outcomes for McNemar (rollout_loop._dump_episodes)
export GACA_STEP_DUMP=$OUT/steps.jsonl        # per-step NLL vs exact entropy, logging only (ray_trainer._dump_steps_impl)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gaca \
    algorithm.gaca.method=$METHOD \
    algorithm.gaca.rel=$REL \
    algorithm.gaca.norm=mean_std \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_BS \
    data.val_batch_size=$VAL_BS \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-6} `# hill-climbing arms raise it via LR` \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO:-4} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True `# 40 GB: Adam state on CPU` \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LP_MICRO:-8} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 `# one GPU (paper recipe: 2)` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_MEM:-0.35} `# recipe: 0.6 of an 80 GB card` \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=${EAGER:-False} \
    actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE:-False} `# True would need enforce_eager (no CUDA graphs)` \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LP_MICRO:-8} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=gaca_term \
    trainer.experiment_name=$NAME \
    trainer.default_local_dir=$OUT/ckpt \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$STEPS \
    trainer.total_training_steps=$STEPS \
    trainer.val_before_train=${VAL_BEFORE:-True} "$@" 2>&1 | tee -a $OUT/train.log
