#!/usr/bin/env python3
"""
Example script demonstrating API usage.
Run this after starting the server to test the system.
"""

import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8000/api"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def check_health():
    """Check API health."""
    print_section("Health Check")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def upload_document(file_path):
    """Upload a document."""
    print_section(f"Uploading Document: {file_path}")
    
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f)}
        response = requests.post(f"{API_BASE}/documents/upload", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
        return data['document_id']
    else:
        print(f"Error: {response.text}")
        return None


def list_documents():
    """List all documents."""
    print_section("Listing Documents")
    response = requests.get(f"{API_BASE}/documents")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total documents: {data['total_count']}")
    for doc in data['documents']:
        print(f"  - {doc['filename']} ({doc['chunks_count']} chunks)")
    return data['documents']


def search_query(query, enable_auto_enrichment=False):
    """Search the knowledge base."""
    print_section(f"Searching: {query}")
    
    payload = {
        "query": query,
        "top_k": 5,
        "enable_auto_enrichment": enable_auto_enrichment
    }
    
    response = requests.post(
        f"{API_BASE}/search",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n📝 Answer:")
        print(f"   {data['answer']}\n")
        
        print(f"📊 Confidence: {data['confidence']:.2%}")
        print(f"✅ Complete: {data['is_complete']}")
        
        if data['sources']:
            print(f"\n📚 Sources ({len(data['sources'])}):")
            for i, source in enumerate(data['sources'][:3], 1):
                print(f"   {i}. {source['document_name']} (relevance: {source['relevance_score']:.2%})")
        
        if data['missing_info']:
            print(f"\n⚠️  Missing Information:")
            for info in data['missing_info']:
                print(f"   - {info}")
        
        if data['enrichment_suggestions']:
            print(f"\n💡 Enrichment Suggestions:")
            for suggestion in data['enrichment_suggestions']:
                print(f"   - [{suggestion['type']}] {suggestion['suggestion']}")
        
        if data['auto_enrichment_applied']:
            print(f"\n🌐 Auto-enrichment applied from: {', '.join(data['auto_enrichment_sources'])}")
        
        return data
    else:
        print(f"Error: {response.text}")
        return None


def rate_answer(query, answer, rating, feedback=None):
    """Rate an answer."""
    print_section(f"Rating Answer: {rating}/5")
    
    payload = {
        "query": query,
        "answer": answer,
        "rating": rating,
        "feedback": feedback
    }
    
    response = requests.post(
        f"{API_BASE}/rate",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")


def get_rating_statistics():
    """Get rating statistics."""
    print_section("Rating Statistics")
    response = requests.get(f"{API_BASE}/ratings/statistics")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total ratings: {data['total_ratings']}")
        print(f"Average rating: {data['average_rating']}/5")
        print(f"Distribution: {data['rating_distribution']}")


def check_enrichment_capabilities():
    """Check enrichment capabilities."""
    print_section("Enrichment Capabilities")
    response = requests.get(f"{API_BASE}/enrichment/capabilities")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def main():
    """Run example workflow."""
    print("\n🚀 RAG Knowledge Base - API Test Script")
    print("=" * 60)
    
    # Check health
    if not check_health():
        print("❌ API is not healthy. Please start the server first.")
        return
    
    # Check enrichment capabilities
    check_enrichment_capabilities()
    
    # Upload sample document
    sample_doc = Path(__file__).parent / "sample_document.txt"
    if sample_doc.exists():
        doc_id = upload_document(sample_doc)
        if doc_id:
            print(f"✅ Document uploaded successfully: {doc_id}")
            time.sleep(2)  # Wait for indexing
    else:
        print(f"⚠️  Sample document not found at {sample_doc}")
    
    # List documents
    list_documents()
    
    # Test queries
    queries = [
        "What is the company's remote work policy?",
        "How many vacation days do employees get?",
        "What is the training budget for professional development?",
        "What are the company values?",
        "What is the policy on cryptocurrency investments?"  # This should be incomplete
    ]
    
    for query in queries:
        result = search_query(query)
        if result:
            # Rate the answer
            rating = 5 if result['confidence'] > 0.7 else 3
            rate_answer(query, result['answer'], rating, "Test feedback")
        time.sleep(1)
    
    # Test with auto-enrichment
    search_query(
        "What is quantum computing?",
        enable_auto_enrichment=True
    )
    
    # Get rating statistics
    get_rating_statistics()
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API.")
        print("   Please make sure the server is running:")
        print("   python -m uvicorn app.main:app --reload\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")

