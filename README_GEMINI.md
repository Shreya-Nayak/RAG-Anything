# RAG-Anything with Gemini Integration

A powerful RAG (Retrieval-Augmented Generation) system that supports multimodal document processing with Google Gemini Flash integration. Process PDFs, DOCX files, images, and more with intelligent querying capabilities.

## 🌟 Features

- **Gemini Integration**: Uses Google Gemini Flash for advanced LLM and vision capabilities
- **Multi-format Support**: Process PDFs, DOCX, TXT, MD, images (PNG, JPG, JPEG)
- **Batch Processing**: Process multiple documents simultaneously
- **Cross-document Queries**: Ask questions that span across multiple documents
- **Interactive Mode**: Chat-like interface for dynamic querying
- **Multimodal Processing**: Handle text, images, tables, and equations
- **Source Tracking**: Know which document provided each answer

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **LibreOffice** (required for DOCX, XLSX, PPTX processing)
   - Windows: Download from [LibreOffice website](https://www.libreoffice.org/download/download/)
   - Ubuntu/Debian: `sudo apt-get install libreoffice`
   - macOS: `brew install libreoffice`
3. **Google Gemini API Key** - Get it from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shreya-Nayak/RAG-Anything.git
   cd RAG-Anything
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_gemini.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Copy the example environment file
   cp env.example .env
   
   # Edit .env and add your Gemini API key
   # GOOGLE_API_KEY=your_gemini_api_key_here
   ```

### Basic Usage

#### Single Document Processing

```bash
# Process a single document
python examples/raganything_example.py document.pdf

# Ask a specific question
python examples/raganything_example.py document.docx --query "What are the main conclusions?"

# Interactive mode
python examples/raganything_example.py document.pdf --interactive
```

#### Batch Processing (Multiple Documents)

```bash
# Process all PDFs in a directory
python examples/batch_raganything_example.py ./documents/

# Process specific files
python examples/batch_raganything_example.py doc1.pdf doc2.docx doc3.txt

# Use glob patterns
python examples/batch_raganything_example.py "./reports/*.pdf" "./data/*.docx"

# Cross-document query
python examples/batch_raganything_example.py ./documents/ --query "Compare the findings across all documents"

# Interactive batch mode
python examples/batch_raganything_example.py ./documents/ --interactive
```

## 📁 Supported File Formats

| Format | Extension | Requirements |
|--------|-----------|--------------|
| PDF | `.pdf` | None (built-in support) |
| Word Document | `.docx` | LibreOffice |
| Text | `.txt` | None |
| Markdown | `.md` | None |
| Images | `.png`, `.jpg`, `.jpeg` | None (Gemini Vision) |
| Excel | `.xlsx` | LibreOffice |
| PowerPoint | `.pptx` | LibreOffice |

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```env
# Required: Gemini API Configuration
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Logging configuration
LOG_DIR=./logs
VERBOSE=false
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5

# Optional: Processing configuration
WORKING_DIR=./rag_storage
MINERU_OUTPUT_DIR=./output
```

### Command Line Options

#### Single Document (`raganything_example.py`)

```bash
python examples/raganything_example.py [file_path] [options]

Options:
  --working_dir, -w    Working directory for RAG storage (default: ./rag_storage)
  --output, -o         Output directory (default: ./output)
  --api-key           Gemini API key (overrides GOOGLE_API_KEY env var)
  --query, -q         Ask a specific question
  --interactive, -i    Enable interactive mode
```

#### Batch Processing (`batch_raganything_example.py`)

```bash
python examples/batch_raganything_example.py [input_paths...] [options]

Options:
  --working_dir, -w    Working directory for batch processing (default: ./rag_storage_batch)
  --output, -o         Output directory (default: ./output_batch)
  --api-key           Gemini API key (overrides GOOGLE_API_KEY env var)
  --query, -q         Ask a question about all documents
  --interactive, -i    Enable interactive mode
  --extensions        Supported file extensions (default: .pdf .docx .txt .md .png .jpg .jpeg)
```

## 💡 Examples

### Example 1: Research Paper Analysis

```bash
# Process multiple research papers
python examples/batch_raganything_example.py ./research_papers/ \
  --query "What are the common methodologies used across these papers?"
```

### Example 2: Financial Report Comparison

```bash
# Compare quarterly reports
python examples/batch_raganything_example.py \
  Q1_report.pdf Q2_report.pdf Q3_report.pdf Q4_report.pdf \
  --query "Compare the revenue trends across all quarters"
```

### Example 3: Interactive Document Exploration

```bash
# Explore documents interactively
python examples/batch_raganything_example.py ./documents/ --interactive
```

## 🔍 Query Examples

When using interactive mode or custom queries, try these examples:

- **Summarization**: "Summarize the key points from all documents"
- **Comparison**: "Compare the main arguments presented in each document"
- **Analysis**: "What are the common themes across these documents?"
- **Specific Questions**: "What recommendations are made for improving performance?"
- **Data Extraction**: "List all the statistics and numbers mentioned"

## 🛠️ Troubleshooting

### Common Issues

1. **LibreOffice Error when processing DOCX**
   ```
   Solution: Install LibreOffice and ensure it's in your system PATH
   Alternative: Convert DOCX to PDF before processing
   ```

2. **Gemini API Key Error**
   ```
   Solution: Verify your API key is correct and set in .env file
   Check: https://aistudio.google.com/app/apikey
   ```

3. **Memory Issues with Large Documents**
   ```
   Solution: Process documents individually or reduce batch size
   Use: Smaller working directories and clear cache between runs
   ```

4. **File Not Found Errors**
   ```
   Solution: Use absolute paths or ensure files exist
   Check: File permissions and supported formats
   ```

### Performance Tips

- **For large documents**: Use smaller batch sizes
- **For multiple documents**: Enable parallel processing where possible  
- **For better accuracy**: Use specific, detailed queries
- **For faster processing**: Disable unnecessary features in config

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit your changes: `git commit -am 'Add some feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [RAG-Anything](https://github.com/OracleLinux/RAG-Anything) - Original framework
- [Google Gemini](https://ai.google.dev/) - AI model integration
- [MinerU](https://github.com/opendatalab/MinerU) - Document parsing capabilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the [troubleshooting section](#-troubleshooting)
2. Search existing [GitHub issues](https://github.com/Shreya-Nayak/RAG-Anything/issues)
3. Create a new issue with detailed information about your problem

---

**Made with ❤️ by [Shreya-Nayak](https://github.com/Shreya-Nayak)**
