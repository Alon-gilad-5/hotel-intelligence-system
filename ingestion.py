"""
Multi-Source Ingestion Pipeline

Ingests hotel/property data from:
- Booking.com dataset (existing)
- Airbnb dataset (new)

Namespaces:
- booking_hotels, booking_reviews
- airbnb_hotels, airbnb_reviews
"""

import os
import sys
import time
import json
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, concat_ws, lit, explode, when, split,
    monotonically_increasing_id, posexplode, trim
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Fix for PySpark on Windows
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# ID prefixes to avoid collisions
BOOKING_PREFIX = "BKG"
AIRBNB_PREFIX = "ABB"


def create_pinecone_index_if_not_exists(index_name: str, dimension: int = 1024):
    """Create Pinecone index if it doesn't exist."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Waiting for index to initialize...")
        time.sleep(30)
    else:
        print(f"Index '{index_name}' already exists.")


def clear_pinecone_namespaces(index_name: str, namespaces: list[str]):
    """Clear all vectors from specified namespaces."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    for namespace in namespaces:
        try:
            print(f"  Clearing namespace '{namespace}'...")
            index.delete(delete_all=True, namespace=namespace)
            print(f"    [OK] {namespace} cleared")
        except Exception as e:
            print(f"    - {namespace}: {e}")


def get_spark_session():
    """Initialize Spark session with memory optimizations."""
    return SparkSession.builder \
        .appName("MultiSourceIngestion") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.maxResultSize", "512m") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .getOrCreate()


# ===========================================
# BOOKING.COM PROCESSING
# ===========================================

def process_booking_hotels(df, limit: int = 5, city_filter: str = None) -> list[Document]:
    """Process Booking.com hotel data into documents.
    
    Args:
        df: Spark DataFrame with hotel data
        limit: Maximum number of hotels to process
        city_filter: If set, only include hotels from this city
    """
    df_hotels = df.select(
        col("hotel_id"),
        col("title"),
        col("description"),
        col("city"),
        col("country"),
        col("review_score"),
        concat_ws(", ", col("most_popular_facilities")).alias("facilities_str")
    ).fillna({
        "description": "",
        "title": "Unknown Hotel",
        "city": "Unknown",
        "country": "Unknown"
    })
    
    # Apply city filter if specified
    if city_filter:
        df_hotels = df_hotels.filter(col("city") == city_filter)
        print(f"    [City Filter] Found {df_hotels.count()} hotels in {city_filter}")
    
    df_hotels = df_hotels.limit(limit)

    rows = df_hotels.collect()
    documents = []

    for row in rows:
        hotel_id = f"{BOOKING_PREFIX}_{row['hotel_id']}"

        content = (
            f"Hotel: {row['title']}. "
            f"Location: {row['city']}, {row['country']}. "
            f"Rating: {row['review_score']}. "
            f"Facilities: {row['facilities_str']}. "
            f"Description: {row['description']}"
        )

        metadata = {
            "source": "booking",
            "hotel_id": hotel_id,
            "original_id": str(row['hotel_id']),
            "title": row['title'],
            "city": row['city'],
            "country": row['country'],
            "rating": float(row['review_score']) if row['review_score'] else 0.0,
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def process_booking_reviews(df, limit: int = 5, city_filter: str = None) -> list[Document]:
    """Process Booking.com reviews into documents.
    
    Args:
        df: Spark DataFrame with hotel data
        limit: Maximum number of hotels to process reviews from
        city_filter: If set, only include reviews from hotels in this city
    """
    # Filter to hotels with reviews
    df_with_reviews = df.filter(col("top_reviews").isNotNull())
    
    # Apply city filter if specified
    if city_filter:
        df_with_reviews = df_with_reviews.filter(col("city") == city_filter)
    
    df_with_reviews = df_with_reviews.limit(limit)

    df_reviews = df_with_reviews.select(
        col("hotel_id"),
        col("title").alias("hotel_title"),
        col("city"),
        col("country"),
        posexplode(col("top_reviews")).alias("review_idx", "review_data")
    ).select(
        col("hotel_id"),
        col("hotel_title"),
        col("city"),
        col("country"),
        col("review_idx"),
        col("review_data.review").alias("review_text"),
        col("review_data.reviewer_name").alias("reviewer")
    ).filter(col("review_text").isNotNull())

    rows = df_reviews.collect()
    documents = []

    for row in rows:
        hotel_id = f"{BOOKING_PREFIX}_{row['hotel_id']}"
        review_id = f"{hotel_id}_R{row['review_idx']}"

        content = f"Review for {row['hotel_title']} in {row['city']}: {row['review_text']}"

        metadata = {
            "source": "booking",
            "review_id": review_id,
            "hotel_id": hotel_id,
            "hotel_name": row['hotel_title'],
            "city": row['city'],
            "country": row['country'],
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


# ===========================================
# AIRBNB PROCESSING
# ===========================================

def parse_airbnb_reviews(reviews_str: str) -> list[str]:
    """
    Parse Airbnb reviews string into individual reviews.
    The format may vary - handle common patterns.
    """
    if not reviews_str or reviews_str.strip() == "":
        return []

    # Try JSON parsing first
    try:
        parsed = json.loads(reviews_str)
        if isinstance(parsed, list):
            # List of review objects or strings
            reviews = []
            for item in parsed:
                if isinstance(item, dict):
                    # Extract review text from dict
                    text = item.get('review') or item.get('text') or item.get('comment') or str(item)
                    reviews.append(text)
                elif isinstance(item, str):
                    reviews.append(item)
            return reviews
        elif isinstance(parsed, str):
            return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: split by common delimiters
    # Check for numbered reviews (1. ... 2. ...)
    if any(f"{i}." in reviews_str for i in range(1, 5)):
        import re
        parts = re.split(r'\d+\.\s*', reviews_str)
        return [p.strip() for p in parts if p.strip()]

    # Split by double newlines or pipe
    for delimiter in ['\n\n', '|', '---']:
        if delimiter in reviews_str:
            parts = reviews_str.split(delimiter)
            return [p.strip() for p in parts if p.strip()]

    # Return as single review if can't parse
    return [reviews_str] if len(reviews_str) > 20 else []


def process_airbnb_hotels(df, limit: int = 5, city_filter: str = None) -> list[Document]:
    """Process Airbnb property data into documents.
    
    Args:
        df: Spark DataFrame with property data
        limit: Maximum number of properties to process
        city_filter: If set, only include properties from this city (matches 'location' field)
    """
    df_properties = df.select(
        col("property_id"),
        col("name"),
        col("listing_name"),
        col("description"),
        col("location"),
        col("country"),
        col("ratings"),
        col("amenities"),
        col("guests"),
        col("price"),
        col("category")
    ).fillna({
        "description": "",
        "name": "Unknown Property",
        "location": "Unknown",
        "country": "Unknown"
    })
    
    # Apply city filter if specified (Airbnb uses 'location' field)
    if city_filter:
        df_properties = df_properties.filter(col("location").contains(city_filter))
        print(f"    [City Filter] Found {df_properties.count()} Airbnb properties in {city_filter}")
    
    df_properties = df_properties.limit(limit)

    rows = df_properties.collect()
    documents = []

    for row in rows:
        # Use property_id or generate one
        original_id = row['property_id'] or str(hash(row['name']))[:10]
        hotel_id = f"{AIRBNB_PREFIX}_{original_id}"

        # Combine name fields
        title = row['listing_name'] or row['name'] or "Airbnb Property"

        # Parse rating
        rating = 0.0
        if row['ratings']:
            try:
                rating = float(row['ratings'].replace(',', '.'))
            except (ValueError, AttributeError):
                pass

        content = (
            f"Property: {title}. "
            f"Location: {row['location']}, {row['country']}. "
            f"Rating: {rating}. "
            f"Guests: {row['guests']}. "
            f"Category: {row['category']}. "
            f"Price: {row['price']}. "
            f"Amenities: {row['amenities']}. "
            f"Description: {row['description']}"
        )

        metadata = {
            "source": "airbnb",
            "hotel_id": hotel_id,
            "original_id": original_id,
            "title": title,
            "city": row['location'] or "",
            "country": row['country'] or "",
            "rating": rating,
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def process_airbnb_reviews(df, limit: int = 5, city_filter: str = None) -> list[Document]:
    """Process Airbnb reviews into documents.
    
    Args:
        df: Spark DataFrame with property data
        limit: Maximum number of properties to process reviews from
        city_filter: If set, only include reviews from properties in this city
    """
    df_with_reviews = df.filter(
        (col("reviews").isNotNull()) &
        (col("reviews") != "")
    ).select(
        col("property_id"),
        col("name"),
        col("listing_name"),
        col("location"),
        col("country"),
        col("reviews")
    )
    
    # Apply city filter if specified
    if city_filter:
        df_with_reviews = df_with_reviews.filter(col("location").contains(city_filter))
    
    df_with_reviews = df_with_reviews.limit(limit)

    rows = df_with_reviews.collect()
    documents = []

    for row in rows:
        original_id = row['property_id'] or str(hash(row['name']))[:10]
        hotel_id = f"{AIRBNB_PREFIX}_{original_id}"
        title = row['listing_name'] or row['name'] or "Airbnb Property"

        # Parse reviews string into individual reviews
        reviews = parse_airbnb_reviews(row['reviews'])

        for idx, review_text in enumerate(reviews):
            if not review_text or len(review_text) < 10:
                continue

            review_id = f"{hotel_id}_R{idx}"

            content = f"Review for {title} in {row['location']}: {review_text}"

            metadata = {
                "source": "airbnb",
                "review_id": review_id,
                "hotel_id": hotel_id,
                "hotel_name": title,
                "city": row['location'] or "",
                "country": row['country'] or "",
            }

            documents.append(Document(page_content=content, metadata=metadata))

    return documents


# ===========================================
# MAIN INGESTION
# ===========================================

def run_ingestion(
        booking_path: str = "data/sampled_booking_data.parquet",
        airbnb_path: str = "data/sampled_airbnb_data.parquet",
        index_name: str = "booking-agent",
        sample_size: int = 5,
        city_filter: str = None,
        clear_existing: bool = True
):
    """
    Main ingestion pipeline for both data sources.

    Args:
        booking_path: Path to Booking.com parquet file
        airbnb_path: Path to Airbnb parquet file
        index_name: Pinecone index name
        sample_size: Number of rows to process from each source
        city_filter: If set, only ingest hotels from this city
        clear_existing: If True, clear existing data before ingesting
    """
    print("=" * 50)
    print("MULTI-SOURCE INGESTION PIPELINE")
    print("=" * 50)
    if city_filter:
        print(f"City Filter: {city_filter}")
    print(f"Sample Size: {sample_size}")

    spark = get_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    # Create Pinecone index
    create_pinecone_index_if_not_exists(index_name, dimension=1024)
    
    # Clear existing data if requested
    if clear_existing:
        print("\nClearing existing Pinecone data...")
        clear_pinecone_namespaces(
            index_name, 
            ["booking_hotels", "booking_reviews", "airbnb_hotels", "airbnb_reviews"]
        )
        time.sleep(2)  # Give Pinecone time to process deletions

    # Initialize embeddings
    print("\nLoading embedding model (BAAI/bge-m3)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    all_docs = {
        "booking_hotels": [],
        "booking_reviews": [],
        "airbnb_hotels": [],
        "airbnb_reviews": []
    }

    # ===========================================
    # Process Booking.com data
    # ===========================================
    if os.path.exists(booking_path):
        print(f"\n[BOOKING] Loading from {booking_path}...")
        booking_df = spark.read.parquet(booking_path)
        booking_df.cache()
        print(f"[BOOKING] Total rows: {booking_df.count()}")

        all_docs["booking_hotels"] = process_booking_hotels(booking_df, sample_size, city_filter)
        all_docs["booking_reviews"] = process_booking_reviews(booking_df, sample_size, city_filter)

        print(f"[BOOKING] Hotels: {len(all_docs['booking_hotels'])}, Reviews: {len(all_docs['booking_reviews'])}")
    else:
        print(f"\n[BOOKING] File not found: {booking_path}")

    # ===========================================
    # Process Airbnb data
    # ===========================================
    if os.path.exists(airbnb_path):
        print(f"\n[AIRBNB] Loading from {airbnb_path}...")
        airbnb_df = spark.read.parquet(airbnb_path)
        airbnb_df.cache()
        print(f"[AIRBNB] Total rows: {airbnb_df.count()}")

        all_docs["airbnb_hotels"] = process_airbnb_hotels(airbnb_df, sample_size, city_filter)
        all_docs["airbnb_reviews"] = process_airbnb_reviews(airbnb_df, sample_size, city_filter)

        print(f"[AIRBNB] Hotels: {len(all_docs['airbnb_hotels'])}, Reviews: {len(all_docs['airbnb_reviews'])}")
    else:
        print(f"\n[AIRBNB] File not found: {airbnb_path}")

    # ===========================================
    # Upload to Pinecone
    # ===========================================
    print("\n" + "=" * 50)
    print("UPLOADING TO PINECONE")
    print("=" * 50)

    for namespace, docs in all_docs.items():
        if docs:
            print(f"\nUpserting {len(docs)} docs to namespace '{namespace}'...")
            PineconeVectorStore.from_documents(
                documents=docs,
                embedding=embeddings,
                index_name=index_name,
                namespace=namespace
            )
            print(f"  [OK] {namespace} complete")
        else:
            print(f"\n  - {namespace}: No documents to upload")

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)

    # Summary
    total_hotels = len(all_docs["booking_hotels"]) + len(all_docs["airbnb_hotels"])
    total_reviews = len(all_docs["booking_reviews"]) + len(all_docs["airbnb_reviews"])
    print(f"\nTotal Hotels: {total_hotels}")
    print(f"Total Reviews: {total_reviews}")
    print(f"Namespaces created: {[k for k, v in all_docs.items() if v]}")

    spark.stop()


def upload_airbnb_to_new_index(
    airbnb_path: str = "data/airbnb_sampled_three_cities.parquet",
    index_name: str = "airbnb-index",
    sample_size: int = 1000,
    city_filter: str = None
):
    """
    Upload Airbnb properties to a new Pinecone index.
    
    Creates "airbnb-index" if it doesn't exist and uploads properties using Spark.
    This is separate from the main ingestion pipeline and doesn't affect existing indexes.
    
    Args:
        airbnb_path: Path to Airbnb parquet file
        index_name: Pinecone index name (default: "airbnb-index")
        sample_size: Number of properties to upload
        city_filter: If set, only include properties from this city
    """
    print("=" * 60)
    print("UPLOADING AIRBNB PROPERTIES TO NEW INDEX")
    print("=" * 60)
    print(f"Index: {index_name}")
    print(f"Source: {airbnb_path}")
    print(f"Sample Size: {sample_size}")
    if city_filter:
        print(f"City Filter: {city_filter}")
    print("=" * 60)
    
    # Create Spark session
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    
    # Create Pinecone index
    create_pinecone_index_if_not_exists(index_name, dimension=1024)
    
    # Initialize embeddings
    print("\nLoading embedding model (BAAI/bge-m3)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # Process Airbnb data
    documents = []
    
    if os.path.exists(airbnb_path):
        print(f"\n[AIRBNB] Loading from {airbnb_path}...")
        airbnb_df = spark.read.parquet(airbnb_path)
        airbnb_df.cache()
        print(f"[AIRBNB] Total rows: {airbnb_df.count()}")
        
        documents = process_airbnb_hotels(airbnb_df, sample_size, city_filter)
        print(f"[AIRBNB] Processed {len(documents)} properties")
    else:
        print(f"\n[AIRBNB] File not found: {airbnb_path}")
        spark.stop()
        return
    
    # Upload to Pinecone in batches to avoid memory issues
    if documents:
        print("\n" + "=" * 60)
        print("UPLOADING TO PINECONE")
        print("=" * 60)
        
        # Smaller batch size for embedding (5 at a time to avoid memory issues)
        embed_batch_size = 5
        total_docs = len(documents)
        total_batches = (total_docs + embed_batch_size - 1) // embed_batch_size
        
        print(f"\nUpserting {total_docs} properties in {total_batches} batches (batch size: {embed_batch_size})...")
        print(f"Index: '{index_name}', Namespace: 'airbnb_hotels'")
        
        try:
            # Initialize Pinecone connection
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(index_name)
            
            # Process in small batches: embed then upload
            for i in range(0, total_docs, embed_batch_size):
                batch = documents[i:i + embed_batch_size]
                batch_num = (i // embed_batch_size) + 1
                print(f"  Batch {batch_num}/{total_batches}: Processing {len(batch)} documents...", end=" ", flush=True)
                
                # Embed this small batch
                texts = [doc.page_content for doc in batch]
                batch_embeddings = embeddings.embed_documents(texts)
                
                # Prepare vectors for upload
                vectors_to_upsert = []
                for j, doc in enumerate(batch):
                    # Use hotel_id as the vector ID, or generate one from metadata
                    vector_id = doc.metadata.get('hotel_id', f"doc_{i+j}")
                    # Include text in metadata for LangChain retrieval
                    metadata_with_text = {**doc.metadata, "text": doc.page_content}
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": batch_embeddings[j],
                        "metadata": metadata_with_text
                    })
                
                # Upload to Pinecone
                index.upsert(vectors=vectors_to_upsert, namespace="airbnb_hotels")
                print("[OK]")
            
            print(f"\n  [OK] All {total_docs} documents uploaded successfully!")
        except Exception as e:
            print(f"\n  [ERROR] Upload failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print("\nNo documents to upload")
    
    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"Uploaded {len(documents)} properties to index '{index_name}'")
    print(f"Namespace: airbnb_hotels")
    print("=" * 60)
    
    spark.stop()


if __name__ == "__main__":
    # Option 1: Upload Airbnb properties to new "airbnb-index"
    upload_airbnb_to_new_index(
        airbnb_path="data/airbnb_sampled_three_cities.parquet",
        index_name="airbnb-index",
        sample_size=100,  # Start with 100 properties to avoid memory issues
        city_filter="Broadbeach"  # Focus on Broadbeach first to get ABB_40458495
    )
    
    # Option 2: Run full ingestion to "booking-agent" index (commented out)
    # run_ingestion(
    #     booking_path="data/sampled_booking_data.parquet",
    #     airbnb_path="data/airbnb_sampled_three_cities.parquet",
    #     index_name="booking-agent",
    #     sample_size=5000,  # Ingest 5000 rows from each source
    #     city_filter=None,  # Include all three cities
    #     clear_existing=True
    # )