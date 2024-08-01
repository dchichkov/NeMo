export NEMO=/opt/NeMo
export NODES=2
export TP=8
export PP=2
export CKPT=/lustre/fsw/coreai_dlalgo_llm/huiyingl/neva/NeMo-Megatron-Launcher-official/launcher_scripts/results/HFopenclip_tile_nevav4_1423k_340b_steps5557_GH2GW3TP8PP2N4/HFopenclip_tile_nevav4_1423k_340b_steps5557_GH2GW3TP8PP2N4_lora_340b_preview10delsys_nv_dpo_12000_LR3e-5MINLR2.999e-5WARMUP0TP8PP8N8/results/checkpoints
export FILE=/opt/NeMo/340b_preview10delsys_nv_dpo_llava_bench.json
#export LOCAL_RANK=0
#export WORLD_SIZE=8
#export RANK=0

export NEMO_TESTING=1

cd /opt/NeMo

python ${NEMO}/examples/multimodal/multimodal_llm/neva/neva_evaluation.py \
   tensor_model_parallel_size=${TP} \
   pipeline_model_parallel_size=${PP} \
   neva_model_file=${CKPT}/nemo_neva.nemo \
   base_model_file=/lustre/fsw/coreai_dlalgo_llm/huiyingl/neva/checkpoints/340b-chat-ex \
   trainer.devices=${TP} \
   trainer.num_nodes=${NODES} \
   prompt_file=/opt/NeMo/llava-bench-in-the-wild/input_llama_format.jsonl \
   inference.media_base_path=/opt/NeMo/llava-bench-in-the-wild/images \
   output_file=${FILE} \
   inference.temperature=0.2 \
   inference.greedy=False \
   inference.add_BOS=False \
   inference.tokens_to_generate=1024

#python /opt/NeMo/examples/multimodal/multimodal_llm/neva/eval/gradio_server.py --model-path /lustre/fsw/coreai_dlalgo_llm/huiyingl/neva/NeMo-Megatron-Launcher-official/launcher_scripts/results/HFopenclip_tile_nevav4_1423k_15b_dpo_steps5557_plain_GH2GW3TP1PP1/HFopenclip_tile_nevav4_1423k_15b_dpo_steps5557_plain_GH2GW3TP1PP1_ft_nemo996a_15b_preview08delsys12000_LR3e-6MINLR2.999e-6WARMUP0TP8PP1/results/checkpoints/nemo_neva.nemo --tp 8 --pp 2
