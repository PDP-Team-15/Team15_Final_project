#!/bin/bash

tune run lora_finetune_single_device --config /home/pdp15/UniPar_AI/llm_finetune/configs/8B_lora_single_device.yaml \
	dataset.packed=False \
	compile=True \
	loss=torchtune.modules.loss.CEWithChunkedOutputLoss \
	enable_activation_checkpointing=True \
	optimizer_in_bwd=False \
	enable_activation_offloading=True \
	optimizer=torch.optim.AdamW \
	clip_grad_norm=1 \
	optimizer.lr=5e-6 \
	tokenizer.max_seq_len=8192 \
	gradient_accumulation_steps=4 \
	epochs=1 \
	batch_size=1
