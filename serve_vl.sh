#!/bin/bash

# Start vllm serve for Qwen/Qwen3-VL-4B-Instruct on port 6006 ( Restrict max model len to prevent OOM ). 

vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --gpu-memory-utilization 0.4 \
    --max-model-len 40000 \
    --port 6006 \
    --allowed-local-media-path $(pwd)
