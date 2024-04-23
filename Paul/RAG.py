# I ran:
# pip3 install sentence_transformers
# in the terminal
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

# Sentences we want to encode. Example:
sentence = ['This framework generates embeddings for each input sentence.']

# Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)

print("embedding", embedding.shape)
# embedding.shape is (1, 768), regardless of the length of the sentence. (cf. it was (1, 384 when I used 'paraphrase-MiniLM-L6-v2'))


##########################################

from datasets import load_dataset
dataset = load_dataset("ms_marco", "v1.1")
print("dataset", dataset)

print("dataset", dataset["train"]["passages"][0])

passages = []
for passage in dataset["train"]["passages"]:
    passages.append(passage['passage_text'])

# Embed the passages using the sentence transformer model
passage_embeddings = model.encode(passages, show_progress_bar=True, batch_size=100)

print("passage_embeddings", passage_embeddings[:2])

# Store the passage embeddings and corresponding passages in a dictionary
embedding_dict = {
    'embeddings': passage_embeddings,
    'passages': passages
}

# Save the embedding dictionary to disk (optional)
import pickle
with open('embedding_dict.pkl', 'wb') as f:
    pickle.dump(embedding_dict, f)


##########################################