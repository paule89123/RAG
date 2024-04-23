import torch

if torch.backends.mps.is_available():  # Check for Apple Silicon GPU availability (requires PyTorch 1.12 or later)
    device = torch.device("mps")
elif torch.cuda.is_available():  # Check for NVIDIA GPU availability
    device = torch.device("cuda")
else:
    device = torch.device("cpu")  # Fall back to CPU

print(f"Using device: {device}")


# I ran:
# pip3 install sentence_transformers
# in the terminal
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device = device)

# Sentences we want to encode. Example:
sentence = ['This framework generates embeddings for each input sentence.']
sentences = ['This framework generates embeddings for each input sentence.', "second sentence"]
nested_sentences = [['This framework generates embeddings for each input sentence.', "second sentence"]]

# Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)

print("embedding", embedding.shape)
# embedding.shape is (1, 768), regardless of the length of the sentence. (cf. it was (1, 384 when I used 'paraphrase-MiniLM-L6-v2'))

embeddings = model.encode(sentences)
print("embeddings", embeddings)

embeddings2 = model.encode(nested_sentences)
print("embeddings2", embeddings2)


##########################################

from datasets import load_dataset
dataset = load_dataset("ms_marco", "v1.1")
print("dataset", dataset)

print("dataset", dataset["train"]["passages"][0]["passage_text"])

passages = []
for passage in dataset["train"]["passages"]:
    for passage in passage['passage_text']:
        passages.append(passage)

print("passages", passages[0])


# Embed the passages using the sentence transformer model
passage_embeddings = model.encode(passages, show_progress_bar=True, batch_size=100, device = device, convert_to_numpy=True)

print("passage_embeddings", passage_embeddings)

# Store the passage embeddings and corresponding passages in a dictionary
embedding_dict = {
    'embeddings': passage_embeddings,
    'passages': passages
}

# Save the embedding dictionary to disk (optional)
import pickle
with open('embedding_dict.pkl', 'wb') as f:
    pickle.dump(embedding_dict, f)


import faiss
index = faiss.IndexFlatL2(embedding_dict['embeddings'].shape[1])
index.add(embedding_dict['embeddings'])

query = "What is the capital of France?"
query_embedding = model.encode([query])
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)
retrieved_passages = [passages[idx] for idx in indices[0]]
print("retrieved_passages", retrieved_passages)