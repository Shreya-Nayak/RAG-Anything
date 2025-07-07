#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process documents with RAGAnything using MinerU parser
2. Perform pure text queries using aquery() method
3. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
4. Handle different types of multimodal content (tables, equations) in queries
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
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    custom_query: str = None,
    interactive: bool = False,
):
    """
    Process document with RAGAnything

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: Gemini API key
        base_url: Optional base URL for API (not used for Gemini)
        working_dir: Working directory for RAG storage
        custom_query: Custom query to ask (optional)
        interactive: Enable interactive query mode
    """
    try:
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
                
                if image_data:
                    # Convert base64 image data to PIL Image
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Construct prompt with system prompt if provided
                    full_prompt = ""
                    if system_prompt:
                        full_prompt += f"{system_prompt}\n\n"
                    full_prompt += prompt
                    
                    response = model.generate_content([full_prompt, image])
                    return response.text
                else:
                    # Fall back to text-only processing
                    return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
            except Exception as e:
                logger.error(f"Gemini Vision API error: {str(e)}")
                raise

        # Define embedding function using Gemini embeddings
        def gemini_embedding_func(texts):
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

        embedding_func = EmbeddingFunc(
            embedding_dim=768,  # Gemini embedding dimension
            max_token_size=8192,
            func=gemini_embedding_func,
        )

        # Initialize RAGAnything with new dataclass structure
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process document
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        logger.info("\n" + "="*50)
        logger.info("DOCUMENT PROCESSING COMPLETED!")
        logger.info("="*50)

        # Handle different query modes
        if custom_query:
            # Single custom query mode
            logger.info(f"\n[Custom Query]: {custom_query}")
            result = await rag.aquery(custom_query, mode="hybrid")
            logger.info(f"Answer: {result}")
            
        elif interactive:
            # Interactive query mode
            logger.info("\n🤖 Interactive Query Mode Started!")
            logger.info("Type your questions below. Type 'quit' or 'exit' to stop.\n")
            
            while True:
                try:
                    query = input("❓ Your question: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        logger.info("👋 Goodbye!")
                        break
                        
                    if not query:
                        print("Please enter a question.")
                        continue
                        
                    logger.info(f"\n[Interactive Query]: {query}")
                    result = await rag.aquery(query, mode="hybrid")
                    logger.info(f"🔍 Answer: {result}\n")
                    
                except KeyboardInterrupt:
                    logger.info("\n👋 Goodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    
        else:
            # Default example queries mode
            logger.info("\nRunning example queries:")

            # 1. Pure text queries using aquery()
            text_queries = [
                "What is the main content of the document?",
                "What are the key topics discussed?",
                "Summarize the document in 3 key points",
            ]

            for query in text_queries:
                logger.info(f"\n[Example Query]: {query}")
                result = await rag.aquery(query, mode="hybrid")
                logger.info(f"Answer: {result}")

            # 2. Example multimodal query
            logger.info("\n[Example Multimodal Query]: Performance analysis")
            multimodal_result = await rag.aquery_with_multimodal(
                "Compare this performance data with any similar results mentioned in the document",
                multimodal_content=[
                    {
                        "type": "table",
                        "table_data": """Method,Accuracy,Processing_Time
                                    RAGAnything,95.2%,120ms
                                    Traditional_RAG,87.3%,180ms
                                    Baseline,82.1%,200ms""",
                        "table_caption": "Performance comparison results",
                    }
                ],
                mode="hybrid",
            )
            logger.info(f"Answer: {multimodal_result}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="Gemini-powered RAG Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
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
        help="Ask a specific question about the document"
    )
    parser.add_argument(
        "--interactive", "-i", 
        action="store_true",
        help="Enable interactive query mode (ask multiple questions)"
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: Gemini API key is required")
        logger.error("Set GOOGLE_API_KEY environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_with_rag(
            args.file_path, 
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

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
