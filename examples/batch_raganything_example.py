#!/usr/bin/env python
"""
Batch processing example for RAGAnything with Gemini integration

This example shows how to:
1. Process multiple documents in a batch
2. Create a unified knowledge base from multiple sources
3. Perform cross-document queries
4. Track document sources in responses
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path
import base64
import io
from PIL import Image
from typing import List, Dict, Any
import glob

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Import Gemini instead of OpenAI
import google.generativeai as genai
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "batch_raganything_example.log"))

    print(f"\nBatch RAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "standard",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                },
            },
            "loggers": {
                "": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": False}
            },
        }
    )

    # Set verbose debug if enabled
    if os.getenv("VERBOSE", "false").lower() in ("true", "1", "yes"):
        set_verbose_debug()


def get_files_from_input(input_paths: List[str], supported_extensions: List[str] = None) -> List[str]:
    """
    Get list of files from input paths (files, directories, or glob patterns)
    
    Args:
        input_paths: List of file paths, directory paths, or glob patterns
        supported_extensions: List of supported file extensions (default: common document types)
    
    Returns:
        List of file paths
    """
    if supported_extensions is None:
        supported_extensions = ['.pdf', '.docx', '.txt', '.md', '.png', '.jpg', '.jpeg']
    
    files = []
    
    for input_path in input_paths:
        path = Path(input_path)
        
        if path.is_file():
            # Single file
            if path.suffix.lower() in supported_extensions:
                files.append(str(path))
            else:
                logger.warning(f"Unsupported file type: {path}")
        
        elif path.is_dir():
            # Directory - find all supported files
            for ext in supported_extensions:
                files.extend(glob.glob(str(path / f"**/*{ext}"), recursive=True))
        
        else:
            # Treat as glob pattern
            matched_files = glob.glob(input_path)
            for file_path in matched_files:
                if Path(file_path).suffix.lower() in supported_extensions:
                    files.append(file_path)
    
    # Remove duplicates and sort
    files = sorted(list(set(files)))
    return files


async def process_batch_with_rag(
    file_paths: List[str],
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = "./rag_storage",
    custom_query: str = None,
    interactive: bool = False,
):
    """
    Process multiple documents with RAGAnything and Gemini

    Args:
        file_paths: List of paths to documents
        output_dir: Output directory for RAG results
        api_key: Gemini API key
        base_url: Optional base URL for API (not used for Gemini)
        working_dir: Working directory for RAG storage
        custom_query: Custom query to ask (optional)
        interactive: Enable interactive query mode
    """
    try:
        logger.info(f"Processing {len(file_paths)} documents in batch mode")
        
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            mineru_parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Configure Gemini API
        genai.configure(api_key=api_key)

        # Define LLM model function using Gemini Flash
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Construct full prompt with system prompt and history
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
            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")
                raise

        # Define vision model function for image processing using Gemini Flash
        def vision_model_func(
            prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs
        ):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Construct full prompt
                full_prompt = ""
                if system_prompt:
                    full_prompt += f"System: {system_prompt}\n\n"
                
                for msg in history_messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    full_prompt += f"{role.capitalize()}: {content}\n"
                
                full_prompt += f"User: {prompt}"
                
                # Prepare content for Gemini
                content_parts = [full_prompt]
                
                if image_data:
                    if isinstance(image_data, str):
                        # Base64 encoded image
                        try:
                            image_bytes = base64.b64decode(image_data)
                            image = Image.open(io.BytesIO(image_bytes))
                            content_parts.append(image)
                        except Exception as e:
                            logger.error(f"Error processing image data: {e}")
                    elif hasattr(image_data, 'read'):
                        # File-like object
                        image = Image.open(image_data)
                        content_parts.append(image)
                
                response = model.generate_content(content_parts)
                return response.text
            except Exception as e:
                logger.error(f"Gemini Vision API error: {str(e)}")
                raise

        # Define embedding function using Gemini
        async def embedding_func(texts):
            try:
                embeddings = []
                for text in texts:
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
                return embeddings
            except Exception as e:
                logger.error(f"Gemini Embedding API error: {str(e)}")
                raise

        # Process all documents
        processed_docs = []
        for i, file_path in enumerate(file_paths):
            logger.info(f"Processing document {i+1}/{len(file_paths)}: {file_path}")
            
            try:
                # Create a unique working directory for each document to track sources
                doc_working_dir = os.path.join(working_dir, f"doc_{i+1}_{Path(file_path).stem}")
                
                doc_config = RAGAnythingConfig(
                    working_dir=doc_working_dir,
                    mineru_parse_method="auto",
                    enable_image_processing=True,
                    enable_table_processing=True,
                    enable_equation_processing=True,
                )
                
                rag = RAGAnything(doc_config)
                
                # Set custom functions
                rag.set_llm_model_func(llm_model_func)
                rag.set_vision_model_func(vision_model_func)
                rag.set_embedding_func(EmbeddingFunc(embedding_func=embedding_func))
                
                # Process the document
                await rag.aprocess_file(file_path)
                
                processed_docs.append({
                    'file_path': file_path,
                    'rag_instance': rag,
                    'working_dir': doc_working_dir,
                    'document_name': Path(file_path).name
                })
                
                logger.info(f"✅ Successfully processed: {file_path}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {str(e)}")
                continue

        if not processed_docs:
            logger.error("No documents were successfully processed!")
            return

        logger.info(f"\n🎉 Successfully processed {len(processed_docs)} documents!")
        logger.info("=" * 50)

        # Now perform queries across all documents
        if custom_query:
            logger.info(f"\n[Custom Query]: {custom_query}")
            await run_cross_document_query(processed_docs, custom_query)
        
        elif interactive:
            logger.info("\n🔄 Entering interactive mode...")
            logger.info("Type 'quit', 'exit', or 'q' to exit")
            
            while True:
                try:
                    user_query = input("\n💬 Enter your question: ").strip()
                    if user_query.lower() in ['quit', 'exit', 'q', '']:
                        break
                    
                    await run_cross_document_query(processed_docs, user_query)
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
        
        else:
            # Run example cross-document queries
            await run_example_queries(processed_docs)

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


async def run_cross_document_query(processed_docs: List[Dict], query: str):
    """Run a query across all processed documents"""
    logger.info(f"\n🔍 Running cross-document query: {query}")
    logger.info("-" * 50)
    
    results = []
    
    for doc_info in processed_docs:
        try:
            rag = doc_info['rag_instance']
            doc_name = doc_info['document_name']
            
            logger.info(f"Querying document: {doc_name}")
            result = await rag.aquery(query, mode="hybrid")
            
            results.append({
                'document': doc_name,
                'result': result,
                'file_path': doc_info['file_path']
            })
            
            logger.info(f"📄 {doc_name}: {result[:200]}{'...' if len(result) > 200 else ''}")
            
        except Exception as e:
            logger.error(f"Error querying {doc_info['document_name']}: {str(e)}")
    
    # Provide a summary if multiple documents were queried
    if len(results) > 1:
        logger.info(f"\n📊 Cross-Document Summary:")
        logger.info("=" * 30)
        
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['document']}: {result['result'][:100]}{'...' if len(result['result']) > 100 else ''}")


async def run_example_queries(processed_docs: List[Dict]):
    """Run example queries across all documents"""
    
    # Cross-document queries
    cross_doc_queries = [
        "What are the main topics covered across all documents?",
        "Compare the key findings or conclusions from each document",
        "What common themes appear in multiple documents?",
        "Summarize the most important information from all documents",
    ]

    for query in cross_doc_queries:
        await run_cross_document_query(processed_docs, query)
        logger.info("\n" + "="*50 + "\n")


def main():
    """Main function to run the batch example"""
    parser = argparse.ArgumentParser(description="Batch Gemini-powered RAG Example")
    parser.add_argument(
        "input_paths", 
        nargs="+", 
        help="Paths to documents, directories, or glob patterns to process"
    )
    parser.add_argument(
        "--working_dir", "-w", 
        default="./rag_storage_batch", 
        help="Working directory path for batch processing"
    )
    parser.add_argument(
        "--output", "-o", 
        default="./output_batch", 
        help="Output directory path"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GOOGLE_API_KEY"),
        help="Gemini API key (defaults to GOOGLE_API_KEY env var)",
    )
    parser.add_argument("--base-url", help="Optional base URL for API (not used for Gemini)")
    
    # Query options
    parser.add_argument(
        "--query", "-q", 
        help="Ask a specific question about all documents"
    )
    parser.add_argument(
        "--interactive", "-i", 
        action="store_true",
        help="Enable interactive query mode (ask multiple questions)"
    )
    parser.add_argument(
        "--extensions", 
        nargs="+",
        default=['.pdf', '.docx', '.txt', '.md', '.png', '.jpg', '.jpeg'],
        help="Supported file extensions (default: .pdf .docx .txt .md .png .jpg .jpeg)"
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: Gemini API key is required")
        logger.error("Set GOOGLE_API_KEY environment variable or use --api-key option")
        return

    # Get all files from input paths
    file_paths = get_files_from_input(args.input_paths, args.extensions)
    
    if not file_paths:
        logger.error("No supported files found in the specified paths!")
        logger.error(f"Supported extensions: {args.extensions}")
        return
    
    logger.info(f"Found {len(file_paths)} files to process:")
    for file_path in file_paths:
        logger.info(f"  - {file_path}")

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_batch_with_rag(
            file_paths, 
            args.output, 
            args.api_key, 
            args.base_url, 
            args.working_dir,
            args.query,
            args.interactive
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("Batch RAGAnything Example")
    print("=" * 40)
    print("Processing multiple documents with multimodal RAG pipeline")
    print("=" * 40)

    main()
