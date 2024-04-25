import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import faiss
import numpy as np

# Custom dataset class
class MSMARCODataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        passages = item['passages']
        target = item['wellFormedAnswers'][0]
        return query, passages, target

# Initialize the models
query_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
doc_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
llm = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Create a FAISS index
def create_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# Generate a response using the LLM
def generate_response(query, passage, max_length=100):
    input_text = f"Query: {query}\nPassage: {passage}\nResponse:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = llm.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Compute the weighted average of responses
def weighted_average(responses, weights):
    weighted_responses = torch.tensor([response * weight for response, weight in zip(responses, weights)])
    return torch.sum(weighted_responses, dim=0)

# Training loop
def train(query_encoder, llm, dataloader, optimizer, criterion, device):
    query_encoder.train()
    llm.train()
    for batch in dataloader:
        queries, passages_list, targets = batch
        queries = list(queries)
        targets = list(targets)

        all_passages = [passage for passages in passages_list for passage in passages['passage_text']]
        all_passage_embeddings = doc_encoder.encode(all_passages)
        index = create_index(all_passage_embeddings)

        batch_loss = 0
        for query, passages, target in zip(queries, passages_list, targets):
            query_embedding = query_encoder.encode([query])
            distances, indices = index.search(query_embedding, k=5)
            weights = nn.functional.softmax(torch.tensor(distances), dim=0)

            responses = []
            for idx in indices[0]:
                passage = all_passages[idx]
                response = generate_response(query, passage)
                responses.append(tokenizer.encode(response, return_tensors='pt'))

            weighted_response = weighted_average(responses, weights)
            target_ids = tokenizer.encode(target, return_tensors='pt')

            loss = criterion(weighted_response.unsqueeze(0), target_ids.unsqueeze(0))
            batch_loss += loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

# Evaluation loop
def evaluate(query_encoder, llm, dataloader, device):
    query_encoder.eval()
    llm.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            queries, passages_list, targets = batch
            queries = list(queries)
            targets = list(targets)

            all_passages = [passage for passages in passages_list for passage in passages['passage_text']]
            all_passage_embeddings = doc_encoder.encode(all_passages)
            index = create_index(all_passage_embeddings)

            batch_loss = 0
            for query, passages, target in zip(queries, passages_list, targets):
                query_embedding = query_encoder.encode([query])
                distances, indices = index.search(query_embedding, k=5)
                weights = nn.functional.softmax(torch.tensor(distances), dim=0)

                responses = []
                for idx in indices[0]:
                    passage = all_passages[idx]
                    response = generate_response(query, passage)
                    responses.append(tokenizer.encode(response, return_tensors='pt'))

                weighted_response = weighted_average(responses, weights)
                target_ids = tokenizer.encode(target, return_tensors='pt')

                loss = criterion(weighted_response.unsqueeze(0), target_ids.unsqueeze(0))
                batch_loss += loss.item()

            total_loss += batch_loss

    return total_loss / len(dataloader)

# Main training and evaluation
def main():
    # Load the dataset
    train_data = load_dataset("ms_marco", "v1.1", split="train")
    val_data = load_dataset("ms_marco", "v1.1", split="validation")

    # Filter out rows with empty wellFormedAnswers, passages, or query
    train_data = train_data.filter(lambda x: len(x['wellFormedAnswers']) > 0 and len(x['passages']) > 0 and len(x['query']) > 0)
    val_data = val_data.filter(lambda x: len(x['wellFormedAnswers']) > 0 and len(x['passages']) > 0 and len(x['query']) > 0)

    # Create DataLoader instances
    train_dataset = MSMARCODataset(train_data)
    val_dataset = MSMARCODataset(val_data)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up the optimizer and loss function
    optimizer = optim.Adam(list(query_encoder.parameters()) + list(llm.parameters()), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Training and evaluation
    num_epochs = 10
    for epoch in range(num_epochs):
        train(query_encoder, llm, train_dataloader, optimizer, criterion, device)
        val_loss = evaluate(query_encoder, llm, val_dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")

if __name__ == '__main__':
    main()