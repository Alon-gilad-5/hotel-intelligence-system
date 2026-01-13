"""Quick test to check what's in the RAG database."""

import sys
import os

# Add paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENTS_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _AGENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = 'booking-agent'

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

print('=' * 60)
print('CHECKING RAG DATABASE')
print('=' * 60)

# Check booking_hotels namespace
print('\n[1] Booking Hotels Namespace:')
try:
    vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, namespace='booking_hotels')
    docs = vs.similarity_search('hotel', k=10)
    if docs:
        for d in docs:
            hotel_id = d.metadata.get('hotel_id', 'N/A')
            title = d.metadata.get('title', 'N/A')
            city = d.metadata.get('city', 'N/A')
            print(f'  - {hotel_id}: {title} ({city})')
    else:
        print('  (No documents found)')
except Exception as e:
    print(f'  Error: {e}')

# Check booking_reviews namespace
print('\n[2] Booking Reviews Namespace (searching for "wifi"):')
try:
    vs2 = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, namespace='booking_reviews')
    docs2 = vs2.similarity_search('wifi', k=5)
    if docs2:
        for d in docs2:
            hotel_id = d.metadata.get('hotel_id', 'N/A')
            content = d.page_content[:100].replace('\n', ' ')
            print(f'  - Hotel: {hotel_id}')
            print(f'    Review: {content}...')
    else:
        print('  (No documents found)')
except Exception as e:
    print(f'  Error: {e}')

# Check airbnb_hotels namespace
print('\n[3] Airbnb Hotels Namespace:')
try:
    vs3 = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, namespace='airbnb_hotels')
    docs3 = vs3.similarity_search('property', k=10)
    if docs3:
        for d in docs3:
            hotel_id = d.metadata.get('hotel_id', 'N/A')
            title = d.metadata.get('title', 'N/A')
            city = d.metadata.get('city', 'N/A')
            print(f'  - {hotel_id}: {title} ({city})')
    else:
        print('  (No documents found)')
except Exception as e:
    print(f'  Error: {e}')

# Check airbnb_reviews namespace
print('\n[4] Airbnb Reviews Namespace:')
try:
    vs4 = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, namespace='airbnb_reviews')
    docs4 = vs4.similarity_search('review', k=5)
    if docs4:
        for d in docs4:
            hotel_id = d.metadata.get('hotel_id', 'N/A')
            content = d.page_content[:100].replace('\n', ' ')
            print(f'  - Hotel: {hotel_id}')
            print(f'    Review: {content}...')
    else:
        print('  (No documents found)')
except Exception as e:
    print(f'  Error: {e}')

print('\n' + '=' * 60)
print('DONE')
print('=' * 60)
