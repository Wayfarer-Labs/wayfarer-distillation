# wayfarer-distillation

ensure torch is 2.6.0 if on Ubuntu 20, because of glibc compatibility for flashattn2
install requirements into your venv, then install flash_attn==2.7.4post1 with the --no-build-isolation flag
download the wan2.1 models and place them in your preferred dir, e.g. `/mnt/data/wan-ai` alongside the text encoders and tokenizers
to generate prompts, download the VideoProM dataset from huggingface
