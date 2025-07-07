#!/usr/bin/env python
"""
Quick test script for RAG-Anything with Gemini integration
"""

import os
import asyncio
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc


def create_test_document():
    """Create a simple test document"""
    content = """
RAG-Anything Test Document

This is a test document for verifying the RAG-Anything system with Gemini integration.

Key Information:
- This system uses Google Gemini Flash for language processing
- It supports multimodal content including text, images, tables, and equations
- The system can process various document formats including PDF, DOCX, and TXT
- Batch processing allows handling multiple documents simultaneously

Technical Features:
1. Advanced document parsing with MinerU
2. Intelligent chunking and embedding
3. Hybrid search capabilities
4. Cross-document querying
5. Source attribution and tracking

Performance Metrics:
- Processing Speed: Fast
- Accuracy: High
- Scalability: Excellent

Conclusion:
RAG-Anything with Gemini provides a robust solution for document processing and intelligent querying.
"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name


async def test_rag_system():
    """Test the RAG system with a simple document"""
    print("🧪 Testing RAG-Anything with Gemini...")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ Error: GOOGLE_API_KEY not set properly")
        print("Please edit .env file and add your real Gemini API key")
        return False
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create test document
        test_file = create_test_document()
        print(f"📄 Created test document: {test_file}")
        
        # Create RAG configuration
        config = RAGAnythingConfig(
            working_dir="./test_rag_storage",
            mineru_parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        
        # Initialize RAG system
        rag = RAGAnything(config)
        
        # Define LLM function
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            full_prompt = ""
            if system_prompt:
                full_prompt += f"System: {system_prompt}\n\n"
            
            for msg in history_messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                full_prompt += f"{role.capitalize()}: {content}\n"
            
            full_prompt += f"User: {prompt}"
            
            response = model.generate_content(full_prompt)
            return response.text
        
        # Define embedding function
        async def embedding_func(texts):
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        
        # Set functions
        rag.set_llm_model_func(llm_model_func)
        rag.set_embedding_func(EmbeddingFunc(embedding_func=embedding_func))
        
        print("⚙️  Processing document...")
        
        # Process document
        await rag.aprocess_file(test_file)
        
        print("✅ Document processed successfully!")
        
        # Test queries
        test_queries = [
            "What is this document about?",
            "What are the key features mentioned?",
            "How would you rate the performance metrics?"
        ]
        
        print("\n🔍 Testing queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            try:
                result = await rag.aquery(query, mode="hybrid")
                print(f"   Answer: {result[:200]}{'...' if len(result) > 200 else ''}")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                return False
        
        # Cleanup
        try:
            os.unlink(test_file)
        except:
            pass
        
        print("\n🎉 All tests passed successfully!")
        print("✅ RAG-Anything with Gemini is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 RAG-Anything Quick Test")
    print("=" * 30)
    
    # Run async test
    success = asyncio.run(test_rag_system())
    
    if success:
        print("\n✅ System is ready to use!")
        print("\n📚 Next steps:")
        print("   python examples/raganything_example.py your_document.pdf")
        print("   python examples/batch_raganything_example.py ./your_documents/")
    else:
        print("\n❌ Test failed. Please check your configuration.")
        print("   1. Verify GOOGLE_API_KEY in .env file")
        print("   2. Check internet connection")
        print("   3. Run: python setup_gemini.py")


if __name__ == "__main__":
    main()
