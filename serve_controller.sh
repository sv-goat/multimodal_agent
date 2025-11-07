#!/bin/bash

vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice \
--tool-call-parser hermes \
--gpu-memory-utilization=0.5 \
--port 8000