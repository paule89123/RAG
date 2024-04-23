import torch
if torch.backends.mps.is_available():  # Check for Apple Silicon GPU availability (requires PyTorch 1.12 or later)
    device = torch.device("mps")
elif torch.cuda.is_available():  # Check for NVIDIA GPU availability
    device = torch.device("cuda")
else:
    device = torch.device("cpu")  # Fall back to CPU
print(f"Using device: {device}")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# Prepare the dataset
class MSMARCODataset(Dataset):
    def __init__(self, dataset, tokenizer_query, tokenizer_passage, tokenizer_answer, max_length):
        self.queries = [q for q in dataset["train"]["query"]]
        self.passages = [p["passage_text"] for p in dataset["train"]["passages"]]
        self.answers = [a["answer"] for a in dataset["train"]["wellFormedAnswers"]]
        self.tokenizer_query = tokenizer_query
        self.tokenizer_passage = tokenizer_passage
        self.tokenizer_answer = tokenizer_answer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        passage = self.passages[idx]
        answer = self.answers[idx]

        query_inputs = self.tokenizer_query(query, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        passage_inputs = self.tokenizer_passage(passage, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        answer_inputs = self.tokenizer_answer(answer, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        return {
            "query_input_ids": query_inputs["input_ids"].squeeze(),
            "query_attention_mask": query_inputs["attention_mask"].squeeze(),
            "passage_input_ids": passage_inputs["input_ids"].squeeze(),
            "passage_attention_mask": passage_inputs["attention_mask"].squeeze(),
            "answer_input_ids": answer_inputs["input_ids"].squeeze(),
            "answer_attention_mask": answer_inputs["attention_mask"].squeeze(),
        }

# Load the MS MARCO dataset
dataset = load_dataset("ms_marco", "v1.1")

# Initialize tokenizers
tokenizer_query = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
tokenizer_passage = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
tokenizer_answer = AutoTokenizer.from_pretrained("t5-base")

# Create the dataset
max_length = 512
dataset = MSMARCODataset(dataset, tokenizer_query, tokenizer_passage, tokenizer_answer, max_length)

# Initialize the query encoder (Q)
query_encoder = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens").to(device)

# Initialize the document encoder (D)
document_encoder = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens").to(device)

# Initialize the generator (P)
generator = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

# Apply LoRA to the query encoder and generator
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q", "v"])
query_encoder = get_peft_model(query_encoder, lora_config)
generator = get_peft_model(generator, lora_config)

# Define the training loop
def train(query_encoder, generator, dataset, epochs, batch_size, learning_rate):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer_query = AdamW(query_encoder.parameters(), lr=learning_rate)
    optimizer_generator = AdamW(generator.parameters(), lr=learning_rate)
    
    query_encoder.train()
    generator.train()
    
    for epoch in range(epochs):
        for batch in dataloader:
            # Move batch tensors to the selected device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Train the query encoder
            query_inputs = {
                "input_ids": batch["query_input_ids"],
                "attention_mask": batch["query_attention_mask"]
            }
            query_embeddings = query_encoder(**query_inputs).last_hidden_state[:, 0, :]
            
            passage_inputs = {
                "input_ids": batch["passage_input_ids"],
                "attention_mask": batch["passage_attention_mask"]
            }
            passage_embeddings = document_encoder(**passage_inputs).last_hidden_state[:, 0, :]
            
            # Compute the contrastive loss
            contrastive_loss = nn.CrossEntropyLoss()(query_embeddings, passage_embeddings)
            
            optimizer_query.zero_grad()
            contrastive_loss.backward()
            optimizer_query.step()
            
            # Train the generator
            answer_inputs = {
                "input_ids": batch["answer_input_ids"],
                "attention_mask": batch["answer_attention_mask"]
            }
            outputs = generator(**answer_inputs, labels=answer_inputs["input_ids"])
            
            generative_loss = outputs.loss
            
            optimizer_generator.zero_grad()
            generative_loss.backward()
            optimizer_generator.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Contrastive Loss: {contrastive_loss.item()}, Generative Loss: {generative_loss.item()}")

# Train the models
epochs = 3
batch_size = 8
learning_rate = 1e-5

train(query_encoder, generator, dataset, epochs, batch_size, learning_rate)