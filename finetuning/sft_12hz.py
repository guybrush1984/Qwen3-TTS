# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SFT fine-tuning for Qwen3-TTS 12Hz models.

Usage as CLI:
    python -m finetuning.sft_12hz --train_jsonl data.jsonl --speaker_name my_speaker

Usage as library:
    from finetuning.sft_12hz import run_sft
    result = run_sft(train_data=..., speaker_names=["spk1", "spk2"], ...)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from finetuning.dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig


FREEZE_STRATEGIES = ("none", "timbre", "timbre_strict")


def run_sft(
    model_path: str,
    train_data: list[dict],
    speaker_names: list[str],
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-6,
    gradient_accumulation_steps: int = 4,
    sub_talker_weight: float = 0.3,
    keep_all_checkpoints: bool = False,
    freeze_strategy: str = "none",
    on_epoch_end: Optional[Callable[[int, str, float], None]] = None,
) -> dict:
    """Core SFT training loop.

    Args:
        model_path: HF model ID or local path to base model.
        train_data: List of dicts with keys: text, audio_codes, ref_audio, (speaker_id).
        speaker_names: Speaker name(s) for the checkpoint config.
        output_dir: Directory for checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        lr: Learning rate.
        gradient_accumulation_steps: Gradient accumulation steps.
        sub_talker_weight: Weight for sub-talker loss (default 0.3).
        keep_all_checkpoints: If False, only keep latest checkpoint.
        freeze_strategy: Controls which parameters are trained:
            "none"          — no freezing, train all params (default).
            "timbre"        — freeze talker LLM, train speaker embedding +
                              full sub-talker (codec_embedding + layers + heads).
            "timbre_strict" — like "timbre" but also freeze the sub-talker's
                              codec embeddings (the feedback path into the LLM),
                              preserving the LLM's input distribution and rhythm.
        on_epoch_end: Callback(epoch, checkpoint_dir, avg_loss) after each epoch save.

    Returns:
        {"loss_history": [...], "speaker_id_map": {name: idx}, "speaker_embeddings": {name: tensor}}
    """
    if freeze_strategy not in FREEZE_STRATEGIES:
        raise ValueError(
            f"Unknown freeze_strategy={freeze_strategy!r}, "
            f"expected one of {FREEZE_STRATEGIES}"
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(model_path)

    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn,
    )

    if freeze_strategy != "none":
        # Freeze the main talker LLM (transformer layers, norms, heads) to
        # preserve timing/pacing/EOS behavior from the base model.
        for param in qwen3tts.model.talker.parameters():
            param.requires_grad = False
        # Unfreeze speaker embedding slot
        qwen3tts.model.talker.model.codec_embedding.weight.requires_grad = True
        # Unfreeze sub-talker (code_predictor) prediction layers
        for param in qwen3tts.model.talker.code_predictor.parameters():
            param.requires_grad = True

        if freeze_strategy == "timbre_strict":
            # Also freeze the sub-talker's codec embeddings — these feed back
            # into the LLM input, so freezing them preserves the LLM's input
            # distribution and thus its rhythm.
            for param in qwen3tts.model.talker.code_predictor.model.codec_embedding.parameters():
                param.requires_grad = False

        trainable = [p for p in qwen3tts.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in qwen3tts.model.parameters())
        accelerator.print(
            f"  freeze_strategy={freeze_strategy!r}: training {n_trainable:,} / "
            f"{n_total:,} params ({100 * n_trainable / n_total:.1f}%)"
        )
        optimizer = AdamW(trainable, lr=lr, weight_decay=0.01)
    else:
        optimizer = AdamW(qwen3tts.model.parameters(), lr=lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader,
    )

    speaker_embeddings: dict[str, torch.Tensor] = {}
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        epoch_losses = []
        for step, batch_data in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = _train_step(
                    model, batch_data, speaker_names, speaker_embeddings,
                    sub_talker_weight,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            if step % 10 == 0:
                accelerator.print(
                    f"  Epoch {epoch} | Step {step} | Loss: {loss_val:.4f}"
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        loss_history.append(avg_loss)
        accelerator.print(f"  Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            ckpt_dir = _save_checkpoint(
                model, accelerator, model_path, output_dir, epoch,
                speaker_names, speaker_embeddings,
                keep_all=keep_all_checkpoints,
            )
            if on_epoch_end:
                on_epoch_end(epoch, ckpt_dir, avg_loss)

    # Build speaker_id_map
    spk_id_map = {name: 3000 + i for i, name in enumerate(speaker_embeddings.keys())}

    return {
        "loss_history": loss_history,
        "speaker_id_map": spk_id_map,
        "speaker_embeddings": speaker_embeddings,
    }


def _train_step(
    model,
    batch_data: dict,
    speaker_names: list[str],
    speaker_embeddings: dict[str, torch.Tensor],
    sub_talker_weight: float,
) -> torch.Tensor:
    """Single training step. Returns the combined loss."""
    input_ids = batch_data["input_ids"]
    codec_ids = batch_data["codec_ids"]
    ref_mels = batch_data["ref_mels"]
    text_embedding_mask = batch_data["text_embedding_mask"]
    codec_embedding_mask = batch_data["codec_embedding_mask"]
    attention_mask = batch_data["attention_mask"]
    codec_0_labels = batch_data["codec_0_labels"]
    codec_mask = batch_data["codec_mask"]

    speaker_embedding = model.speaker_encoder(
        ref_mels.to(model.device).to(model.dtype)
    ).detach()

    # Track speaker embeddings for checkpoint baking
    if "speaker_ids" in batch_data:
        for i, spk_id in enumerate(batch_data["speaker_ids"]):
            if spk_id not in speaker_embeddings:
                speaker_embeddings[spk_id] = speaker_embedding[i : i + 1]
    elif not speaker_embeddings:
        speaker_embeddings[speaker_names[0]] = speaker_embedding

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    # Text embedding with projection (needed for 0.6B model)
    input_text_embedding = model.talker.model.text_embedding(input_text_ids)
    if hasattr(model.talker, "text_projection"):
        input_text_embedding = model.talker.text_projection(input_text_embedding)
    input_text_embedding = input_text_embedding * text_embedding_mask

    input_codec_embedding = (
        model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    )
    input_codec_embedding[:, 6, :] = speaker_embedding

    input_embeddings = input_text_embedding + input_codec_embedding

    # Add sub-talker codec layers 1-15 to input embeddings.
    # Reverted removal from PR #178 — removing these caused speech to be too fast.
    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
            codec_ids[:, :, i]
        )
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    # Main AR forward — don't pass labels= (HF shifts internally, causing
    # double-shift). See: github.com/QwenLM/Qwen3-TTS/issues/179
    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        output_hidden_states=True,
    )

    logits = outputs.logits
    loss_main = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        codec_0_labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]

    sub_talker_logits, _ = model.talker.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden_states,
    )
    sub_talker_loss = F.cross_entropy(
        sub_talker_logits.reshape(-1, sub_talker_logits.size(-1)),
        talker_codec_ids[:, 1:].reshape(-1),
    )

    return loss_main + sub_talker_weight * sub_talker_loss


def _save_checkpoint(
    model,
    accelerator,
    base_model_path: str,
    output_dir: str,
    epoch: int,
    speaker_names: list[str],
    speaker_embeddings: dict[str, torch.Tensor],
    keep_all: bool = False,
) -> str:
    """Save a checkpoint with baked speaker embeddings. Returns checkpoint dir."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")

    # Remove previous checkpoint if not keeping all
    if not keep_all and epoch > 0:
        prev = os.path.join(output_dir, f"checkpoint-epoch-{epoch - 1}")
        if os.path.exists(prev):
            shutil.rmtree(prev)

    shutil.copytree(base_model_path, ckpt_dir, dirs_exist_ok=True)

    # Update config with speaker info
    config_file = os.path.join(ckpt_dir, "config.json")
    with open(config_file) as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    spk_id_map = {name: 3000 + i for i, name in enumerate(speaker_embeddings.keys())}
    talker_config["spk_id"] = spk_id_map
    talker_config["spk_is_dialect"] = {name: False for name in spk_id_map}
    config_dict["talker_config"] = talker_config
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # Save model weights with baked speaker embeddings
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {
        k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()
    }

    # Drop speaker_encoder weights (not needed for inference)
    for k in [k for k in state_dict if k.startswith("speaker_encoder")]:
        del state_dict[k]

    # Bake speaker embeddings into codec_embedding weight
    weight = state_dict["talker.model.codec_embedding.weight"]
    for spk_name, emb in speaker_embeddings.items():
        idx = spk_id_map[spk_name]
        weight[idx] = emb[0].detach().to(weight.device).to(weight.dtype)
        print(f"    Baked {spk_name} at index {idx}")

    save_file(state_dict, os.path.join(ckpt_dir, "model.safetensors"))
    return ckpt_dir


def train():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--freeze_strategy", type=str, default="none",
                        choices=FREEZE_STRATEGIES,
                        help="none=train all, timbre=freeze LLM, timbre_strict=also freeze feedback embeddings")
    args = parser.parse_args()

    with open(args.train_jsonl) as f:
        train_data = [json.loads(line) for line in f]

    result = run_sft(
        model_path=args.init_model_path,
        train_data=train_data,
        speaker_names=[args.speaker_name],
        output_dir=args.output_model_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_strategy=args.freeze_strategy,
    )

    print(f"Training complete. Loss history: {result['loss_history']}")


if __name__ == "__main__":
    train()
