import os
import logging
from transformers import CLIPModel, CLIPTokenizer
from pinecone import Pinecone, Index
from nltk.tokenize import sent_tokenize
import nltk
import torch
import time
import random
from dotenv import load_dotenv
import re
from pinecone import Pinecone, ServerlessSpec, Index


# Download and specify the nltk data directory
nltk.download('punkt', download_dir='/Users/nishitamatlani/nltk_data')
nltk.data.path.append('/Users/nishitamatlani/nltk_data')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Load credentials from .env
PINECONE_API_KEY = "pcsk_6K3Vu8_56paG1Doi1xgV1FKmXS8FfYJbe1p7HAp92c6QnV4pMCEhJ32otRXqvryzNzGdsQ"
PINECONE_ENVIRONMENT = "vo2w95e.svc.aped-4627-b74a.pinecone.io"
TEXT_INDEX_NAME = "test"  # Name of the Pinecone index
DIMENSION = 512  # CLIP produces 512-dimensional embeddings
METRIC = "cosine"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if it does not exist
if TEXT_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=TEXT_INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to Pinecone index
text_index = pc.Index(TEXT_INDEX_NAME)

# Check if the index is accessible
try:
    _log.info("Connected to text index: %s", text_index.describe_index_stats())
except Exception as e:
    _log.error(f"Failed to connect to Pinecone index: {e}")

# Initialize Hugging Face CLIP model and tokenizer
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

# Function to clean text before embedding
def clean_text(text):
    """Cleans the input text by removing unwanted characters and formatting."""
    try:
        # Replace newlines with a space
        text = text.replace('\n', ' ')
        # Remove non-alphanumeric characters (except for basic punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        _log.error(f"Error cleaning text: {e}")
        return text

# Function to generate CLIP embeddings for text
def get_clip_embedding(text):
    """Generates a 512-dimensional embedding for the input text using CLIP."""
    try:
        inputs = clip_tokenizer(text, return_tensors="pt", truncation=True, max_length=77)
        with torch.no_grad():
            embeddings = clip_model.get_text_features(**inputs)
        # Normalize embeddings
        normalized_embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return normalized_embeddings.squeeze().tolist()
    except Exception as e:
        _log.error(f"Error generating embedding for text: {e}")
        return None

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

# Function to upload text embeddings to Pinecone with retry logic
def upload_to_pinecone_with_retry(embeddings, index, batch_size=10, max_retries=3):
    """Uploads embeddings to Pinecone with retry logic."""
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        retries = 0
        while retries < max_retries:
            try:
                index.upsert(vectors=[{
                    "id": entry["id"],
                    "values": entry["embedding"],
                    "metadata": entry["metadata"]
                } for entry in batch])
                _log.info(f"Uploaded batch {i // batch_size + 1} to Pinecone.")
                break
            except Exception as e:
                retries += 1
                time.sleep(2 + random.uniform(0, 1))  # Exponential backoff
                _log.error(f"Retry {retries}/{max_retries} for batch {i // batch_size + 1}: {e}")
        else:
            _log.error(f"Failed to upload batch {i // batch_size + 1} after {max_retries} retries.")

# Function to process and embed text from a local file
def process_text_file(file_path):
    """Reads a text file, cleans its content, chunks it, generates embeddings, and uploads them to Pinecone."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            full_text = file.read()

        # Clean the text
        cleaned_text = clean_text(full_text)

        # Chunk cleaned text using the sentence-level chunking strategy
        chunks = chunk_text(cleaned_text, max_tokens=100)  # Adjust token limit as needed
        embeddings = []

        for idx, chunk in enumerate(chunks):
            embedding = get_clip_embedding(chunk)
            if embedding:
                embeddings.append({
                    "id": f"text-chunk-{idx}",
                    "embedding": embedding,
                    "metadata": {"chunk_id": idx, "text": chunk}
                })
                _log.info(f"Processed chunk {idx}: {chunk[:30]}...")

        upload_to_pinecone_with_retry(embeddings, text_index)

        _log.info(f"Successfully processed {len(chunks)} chunks from {file_path}.")
    except Exception as e:
        _log.error(f"Error processing file {file_path}: {e}")

def main():
    """Main function to process a text file."""
    text_file_path = "/Users/nishitamatlani/Documents/final_project/web_scraping/scraped_data.txt"  # Replace with the path to your text file
    process_text_file(text_file_path)

if __name__ == "__main__":
    main()
