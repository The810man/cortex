# The brain's night school: a machine with the graphics card, in which the thinking model learns
# from the brain's own day (src/cortex/train.py). Started by the launcher while the brain sleeps
# and the model server has let go of the card. Large: it holds PyTorch and Unsloth.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir unsloth trl peft datasets gguf safetensors sentencepiece
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && pip install --no-cache-dir -e /opt/llama.cpp/gguf-py
# The model server no longer loads adapters beside a model: what was learned is folded into a copy
# of the model, which llama.cpp's quantizer then makes as small as the model the server runs.
RUN apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/* \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build /opt/llama.cpp/build --target llama-quantize -j 8
# pip brings a torchaudio built for another torch, which fails as it loads; nothing here hears.
RUN pip uninstall -y torchaudio
ENV HF_HOME=/models
WORKDIR /work
