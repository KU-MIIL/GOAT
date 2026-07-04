import os
import json
import logging
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--log_step", type=int, default=10)
    return parser.parse_args()


def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def load_data(train_file):
    with open(train_file, "r") as f:
        data = json.load(f)
    examples = []
    for pair in data:
        if isinstance(pair, list) and len(pair) == 2:
            query, doc = pair
            examples.append(InputExample(texts=[query, doc]))
    return examples


def info_nce_loss(emb_q, emb_d, temperature=0.05):
    emb_q = F.normalize(emb_q, dim=1)
    emb_d = F.normalize(emb_d, dim=1)
    logits = torch.matmul(emb_q, emb_d.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    return F.cross_entropy(logits, labels)


def train(model, train_examples, output_dir, batch_size, epochs, lr, log_step):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        logging.info(f"Epoch {epoch + 1} started.")
        for batch in dataloader:
            features, _ = batch
            features = [
                {k: v.to(device) for k, v in feat.items()}
                for feat in features
            ]

            output_q = model(features[0])["sentence_embedding"]
            output_d = model(features[1])["sentence_embedding"]
            loss = info_nce_loss(output_q, output_d)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            total_loss += loss.item()

            if step % log_step == 0:
                logging.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}\n")

    save_path = os.path.join(output_dir, "retriever_model")
    model.save(save_path)
    logging.info(f"Model saved to {save_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training.log")
    setup_logger(log_path)

    logging.info("Initializing training...")
    logging.info(f"Use pytorch device_name: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logging.info(f"Load pretrained SentenceTransformer: {args.model_name}")

    model = SentenceTransformer(args.model_name)
    train_examples = load_data(args.train_file)
    logging.info(f"Loaded {len(train_examples)} training examples.")

    train(
        model=model,
        train_examples=train_examples,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        log_step=args.log_step
    )


if __name__ == "__main__":
    main()
