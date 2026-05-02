import argparse
import json
import math
import os
import torch
from torch.utils.data import DataLoader, SequentialSampler
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Training data JSONL file")
parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--output-dir", default=None, help="Output directory (default: auto-generated)")
parser.add_argument("--lora", action="store_true", default=True, help="Use LoRA (default)")
parser.add_argument("--no-lora", dest="lora", action="store_false", help="Use full fine-tuning instead of LoRA")
parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--grad-accum", type=int, default=5)
parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-4 for LoRA, 2e-5 for full)")
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--checkpoint-every-n-prompts", type=int, default=50)
parser.add_argument("--seed", type=int, default=55)
args = parser.parse_args()

if args.lr is None:
    args.lr = 1e-4 if args.lora else 2e-5

with open(args.input) as f:
    raw_rows = [json.loads(line) for line in f]

print(f"Training data: {args.input} ({len(raw_rows)} examples)")
print(f"Model: {args.model}")
print(
    f"Mode: {'LoRA (r={}, alpha={})'.format(args.lora_r, args.lora_alpha) if args.lora else 'Full fine-tuning'}"
)
print(f"Learning rate: {args.lr}")

rows = []
for item in raw_rows:
    if "adversarial" in item:
        prompt_text = item["adversarial"]
        response_text = item["safe_response"]
        source = item.get("source", "adversarial")
    else:
        prompt_text = item["prompt"]
        response_text = item["response"]
        source = item.get("source", "control")

    rows.append({
        "prompt": [{"role": "user", "content": prompt_text}],
        "completion": [{"role": "assistant", "content": response_text}],
        "source": source,
    })

train_dataset = Dataset.from_list(rows)

effective_batch_size = args.batch_size * args.grad_accum
total_steps = math.ceil(len(raw_rows) / effective_batch_size)

if args.checkpoint_every_n_prompts % effective_batch_size != 0:
    print(
        f"WARNING: --checkpoint-every-n-prompts ({args.checkpoint_every_n_prompts}) "
        f"is not divisible by effective batch size ({effective_batch_size}). "
        f"Checkpoints will not align exactly."
    )

save_steps = max(1, args.checkpoint_every_n_prompts // effective_batch_size)

print(f"Effective batch size: {effective_batch_size}")
print(f"Total training steps: {total_steps}")
print(f"Saving checkpoint every {save_steps} steps ({save_steps * effective_batch_size} prompts)")

# Useful warning for mixed runs:
# to preserve exact 50/50 prefixes at checkpoints, checkpoint size should be even.
if "mixed" in os.path.basename(args.input):
    if args.checkpoint_every_n_prompts % 2 != 0:
        print(
            "WARNING: mixed dataset with odd checkpoint-every-n-prompts. "
            "Exact 50/50 hard/control prefix only holds for even prompt counts."
        )
    if effective_batch_size % 2 != 0:
        print(
            "WARNING: mixed dataset with odd effective batch size. "
            "Per-step batches will not be exactly 50/50, though prefixes over even numbers still are."
        )

print()

if args.output_dir:
    output_dir = args.output_dir
else:
    model_short = args.model.split("/")[-1]
    data_short = os.path.basename(args.input).replace(".jsonl", "")
    mode_short = f"lora_r{args.lora_r}" if args.lora else "full"
    # Set CHECKPOINT_ROOT in your environment to override (e.g., /your/scratch/dir).
    checkpoint_root = os.environ.get("CHECKPOINT_ROOT", "./checkpoints")
    output_dir = f"{checkpoint_root}/{model_short}/{data_short}/{mode_short}"

peft_config = None
if args.lora:
    from peft import LoraConfig

    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    peft_config = LoraConfig(**lora_kwargs)

training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    model_init_kwargs={"dtype": torch.bfloat16},
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=args.max_length,
    save_strategy="steps",
    save_steps=save_steps,
    logging_steps=1,
    report_to="none",
    seed=args.seed,
)


class OrderedSFTTrainer(SFTTrainer):
    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )


trainer = OrderedSFTTrainer(
    model=args.model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

print(f"Output directory: {output_dir}")
print("Training order is preserved from the JSONL file.")
print("Starting training...")
print()

trainer.train()
trainer.save_model(os.path.join(output_dir, "final"))
print(f"\nTraining complete. Final model saved to {output_dir}/final")