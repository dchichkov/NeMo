export NEMO=/opt/NEMO
#export TP=8
#export PP=2
#export LOCAL_RANK=0
#export WORLD_SIZE=8
#export RANK=0
export NEMO_TESTING=1

# /lustre/fsw/coreai_dlalgo_llm/huiyingl/neva/NeMo-Megatron-Launcher-official/launcher_scripts/results/HFopenclip_tile_nevav4_1423k_340b_steps5557_GH2GW3TP8PP2N4/HFopenclip_tile_nevav4_1423k_340b_steps5557_GH2GW3TP8PP2N4_lora_340b_preview10delsys_nv_dpo_12000_LR3e-5MINLR2.999e-5WARMUP0TP8PP8N8/results/checkpoints/nemo_neva.nemo

cd /opt/NeMo 

python /opt/NeMo/examples/multimodal/multimodal_llm/neva/eval/gradio_server.py --model-path /lustre/fsw/coreai_dlalgo_llm/huiyingl/neva/NeMo-Megatron-Launcher-official/launcher_scripts/results/HFopenclip_tile_nevav4_1423k_340b_steps5557_GH2GW3TP8PP2N4/HFopenclip_tile_nevav4_1423k_340b_steps5557_GH2GW3TP8PP2N4_lora_340b_preview10delsys_nv_dpo_12000_LR3e-5MINLR2.999e-5WARMUP0TP8PP8N8/results/checkpoints/nemo_neva.nemo --model-base /lustre/fsw/coreai_dlalgo_llm/huiyingl/neva/checkpoints/340b-chat-ex --tp 8 --pp 2

#python /opt/NeMo/examples/multimodal/multimodal_llm/neva/eval/gradio_server.py --model-path /lustre/fsw/coreai_dlalgo_llm/huiyingl/neva/NeMo-Megatron-Launcher-official/launcher_scripts/results/HFopenclip_tile_nevav4_1423k_15b_dpo_steps5557_plain_GH2GW3TP1PP1/HFopenclip_tile_nevav4_1423k_15b_dpo_steps5557_plain_GH2GW3TP1PP1_ft_nemo996a_15b_preview08delsys12000_LR3e-6MINLR2.999e-6WARMUP0TP8PP1/results/checkpoints/nemo_neva.nemo --tp 8 --pp 2
