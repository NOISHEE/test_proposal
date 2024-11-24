import pinecone
from transformers import CLIPTokenizer, CLIPModel
import torch
from nltk.tokenize import sent_tokenize
import nltk

# Download NLTK sentence tokenizer (only required for the first run)
nltk.download('punkt')

# Pinecone API and Environment setup
PINECONE_API_KEY = "pcsk_6K3Vu8_56paG1Doi1xgV1FKmXS8FfYJbe1p7HAp92c6QnV4pMCEhJ32otRXqvryzNzGdsQ"
PINECONE_ENVIRONMENT = "vo2w95e.svc.aped-4627-b74a.pinecone.io"
TEXT_INDEX_NAME = "test"  # Name of the Pinecone index
DIMENSION = 512  # CLIP produces 512-dimensional embeddings
METRIC = "cosine"

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Create Pinecone index if it does not exist
if TEXT_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=TEXT_INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC
    )

# Connect to the Pinecone index
text_index = pinecone.Index(TEXT_INDEX_NAME)

# Check if the index is accessible
try:
    print("Connected to text index:")
    print(text_index.describe_index_stats())
except Exception as e:
    print(f"Failed to connect to Pinecone index: {e}")

# Load Hugging Face CLIP model and tokenizer
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

# Function to get CLIP embeddings
def get_clip_embedding(text):
    """
    Generates a 512-dimensional embedding for the input text using CLIP.
    """
    inputs = clip_tokenizer(text, return_tensors="pt", truncation=True, max_length=77)
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
    # Normalize the embedding
    normalized_embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return normalized_embeddings.squeeze().tolist()

# Function to chunk text into smaller parts, preserving sentence boundaries
def chunk_text(text, max_tokens=500):
    """
    Splits text into chunks at sentence boundaries, ensuring each chunk
    does not exceed the max_tokens limit.
    """
    # Split text into sentences
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_chunk_token_count = 0

    for sentence in sentences:
        # Count the tokens in the sentence
        sentence_token_count = len(sentence.split())

        # If adding this sentence exceeds the token limit, finalize the current chunk
        if current_chunk_token_count + sentence_token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_chunk_token_count = sentence_token_count
        else:
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_chunk_token_count += sentence_token_count

    # Add the last chunk if there is any remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Read text data
with open("scraped_data.txt", "r", encoding="utf-8") as file:
    full_text = file.read()

# Chunk text using the provided chunking logic
chunks = chunk_text(full_text, max_tokens=100)  # Adjust token limit as needed

# Store in Pinecone
for i, chunk in enumerate(chunks):
    embedding = get_clip_embedding(chunk)  # Generate CLIP embedding
    metadata = {"chunk_id": i, "text": chunk}  # Metadata for retrieval
    try:
        text_index.upsert([(f"chunk-{i}", embedding, metadata)])
        print(f"Chunk {i} stored successfully.")
    except Exception as e:
        print(f"Failed to store chunk {i}: {e}")

print(f"Stored {len(chunks)} chunks in Pinecone.")
