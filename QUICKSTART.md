# 🚀 Quick Start Guide: RAG-Anything with Gemini

Welcome to your enhanced RAG-Anything repository with Google Gemini integration! This guide will get you up and running in just a few minutes.

## 📋 Prerequisites Checklist

Before you start, make sure you have:

- [ ] **Python 3.8+** installed
- [ ] **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
- [ ] **LibreOffice** installed (for DOCX/XLSX/PPTX support)
- [ ] **Git** configured with your GitHub account

## ⚡ Quick Setup (5 minutes)

### 1. Clone and Setup
```bash
# The repository is already cloned to your desktop
cd "c:\Users\SH40178929\Desktop\RAG\RAG-Anything"

# Run the automated setup
python setup_gemini.py
```

### 2. Configure API Key
Edit `.env` file and replace `your_gemini_api_key_here` with your actual Gemini API key:
```env
GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Test Installation
```bash
python test_gemini.py
```

## 🎯 Ready-to-Run Examples

### Single Document Processing
```bash
# Basic processing
python examples/raganything_example.py document.pdf

# Ask specific questions
python examples/raganything_example.py document.docx --query "What are the main conclusions?"

# Interactive chat mode
python examples/raganything_example.py document.pdf --interactive
```

### Multi-Document Processing
```bash
# Process all documents in a folder
python examples/batch_raganything_example.py ./documents/

# Process specific files
python examples/batch_raganything_example.py doc1.pdf doc2.docx doc3.txt

# Cross-document analysis
python examples/batch_raganything_example.py ./reports/ --query "Compare findings across all reports"

# Interactive multi-document chat
python examples/batch_raganything_example.py ./documents/ --interactive
```

## 📁 Test with Sample Documents

Create a test folder and try it out:

```bash
# Create test directory
mkdir test_documents

# Add some sample files (PDF, DOCX, TXT)
# Then run:
python examples/batch_raganything_example.py ./test_documents/ --interactive
```

## 🔧 Supported File Types

| Format | Extension | Status | Requirements |
|--------|-----------|--------|--------------|
| ✅ PDF | `.pdf` | Full Support | None |
| ✅ Text | `.txt`, `.md` | Full Support | None |
| ✅ Images | `.png`, `.jpg`, `.jpeg` | Full Support | None (Gemini Vision) |
| ⚠️ Word | `.docx` | Requires LibreOffice | LibreOffice |
| ⚠️ Excel | `.xlsx` | Requires LibreOffice | LibreOffice |
| ⚠️ PowerPoint | `.pptx` | Requires LibreOffice | LibreOffice |

## 🚨 Troubleshooting

### Common Issues and Solutions

**1. "GOOGLE_API_KEY not set" Error**
```bash
# Edit .env file and add your real API key
# Get one from: https://aistudio.google.com/app/apikey
```

**2. LibreOffice Error with DOCX files**
```bash
# Install LibreOffice from: https://www.libreoffice.org/download/download/
# Alternative: Convert DOCX to PDF first
```

**3. Import or Module Errors**
```bash
# Reinstall dependencies
pip install -r requirements_gemini.txt
```

**4. API Rate Limits**
```bash
# Add delays between requests or use smaller batch sizes
# Gemini has generous free tier limits
```

## 📚 Advanced Usage

### Batch Processing Patterns
```bash
# Process by file type
python examples/batch_raganything_example.py "*.pdf" "*.docx"

# Process recursively
python examples/batch_raganything_example.py ./project_docs/ --extensions .pdf .md .txt

# Custom query across all documents
python examples/batch_raganything_example.py ./research/ --query "What are the key methodologies mentioned?"
```

### Environment Configuration
```env
# .env file options
GOOGLE_API_KEY=your_key_here
LOG_DIR=./logs              # Where to save logs
VERBOSE=true                # Enable debug logging
WORKING_DIR=./rag_storage   # RAG storage location
```

## 🎉 What's New in This Version

- ✨ **Google Gemini Integration**: Advanced LLM and vision capabilities
- 🔄 **Batch Processing**: Handle multiple documents simultaneously  
- 🔍 **Cross-Document Queries**: Ask questions spanning multiple files
- 💬 **Interactive Mode**: Chat-like interface for dynamic exploration
- 📊 **Source Tracking**: Know which document provided each answer
- 🛠️ **Enhanced Setup**: Automated configuration and testing scripts

## 🤝 Need Help?

1. **Check the logs**: `./logs/raganything_example.log`
2. **Run diagnostics**: `python setup_gemini.py`
3. **Test connection**: `python test_gemini.py`
4. **Read full docs**: [README_GEMINI.md](README_GEMINI.md)
5. **Create an issue**: [GitHub Issues](https://github.com/Shreya-Nayak/RAG-Anything/issues)

## 🌟 Repository Status

Your repository at `https://github.com/Shreya-Nayak/RAG-Anything` is now:

- ✅ **Updated** with latest Gemini integration
- ✅ **Pushed** to GitHub with all changes
- ✅ **Ready** for production use
- ✅ **Documented** with comprehensive guides

---

**Happy RAG-ing! 🎯**

*Made with ❤️ by [Shreya-Nayak](https://github.com/Shreya-Nayak)*
