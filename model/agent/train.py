import argparse
import logging
import os
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    TrainerCallback, TrainerControl, TrainerState
)
from datasets import load_dataset
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA3-8B with LoRA on API parameter generation")

    parser.add_argument("--train_file", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model name or path")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    return parser.parse_args()


def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # file output
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            logging.info(f"Step {state.global_step} | Epoch {state.epoch:.2f} | Loss: {logs['loss']:.4f}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # logging setup
    log_file_path = os.path.join(args.output_dir, "training.log")
    setup_logger(log_file_path)
    logging.info("Logger initialized. Starting training...")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    raw_dataset = load_dataset("json", data_files={"train": args.train_file})["train"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        save_total_limit=3,
        save_strategy="epoch",
        logging_steps=100,
        logging_dir=f"{args.output_dir}/logs",
        report_to="none"
    )

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    print_trainable_parameters(model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        callbacks=[LossLoggerCallback()]
    )

    trainer.train()

    # save final adapter
    final_save_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_save_path)
    logging.info(f"Final LoRA adapter model saved to: {final_save_path}")


if __name__ == '__main__':
    main()

