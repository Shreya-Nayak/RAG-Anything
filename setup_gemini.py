#!/usr/bin/env python
"""
Setup script for RAG-Anything with Gemini integration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} - OK")
    return True


def check_libreoffice():
    """Check if LibreOffice is installed"""
    try:
        # Try to find LibreOffice executable
        if shutil.which("libreoffice") or shutil.which("soffice"):
            print("✅ LibreOffice - Found")
            return True
        else:
            print("⚠️  LibreOffice - Not found (required for DOCX/XLSX/PPTX processing)")
            print("   Install from: https://www.libreoffice.org/download/download/")
            return False
    except Exception:
        print("⚠️  LibreOffice - Could not check (required for DOCX/XLSX/PPTX processing)")
        return False


def install_dependencies():
    """Install Python dependencies"""
    try:
        print("📦 Installing Python dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_gemini.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def setup_environment():
    """Set up environment file"""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        try:
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from env.example")
            print("🔑 Please edit .env and add your GOOGLE_API_KEY")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    else:
        # Create basic .env file
        env_content = """# Gemini API Configuration
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Logging configuration
LOG_DIR=./logs
VERBOSE=false

# Optional: Processing configuration
WORKING_DIR=./rag_storage
MINERU_OUTPUT_DIR=./output
"""
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            print("✅ Created .env file")
            print("🔑 Please edit .env and add your GOOGLE_API_KEY")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False


def create_directories():
    """Create necessary directories"""
    directories = ["logs", "rag_storage", "output", "rag_storage_batch", "output_batch"]
    
    for dir_name in directories:
        try:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
        except Exception as e:
            print(f"⚠️  Could not create directory {dir_name}: {e}")


def test_gemini_connection():
    """Test Gemini API connection"""
    print("\n🧪 Testing Gemini API connection...")
    
    # Check if API key is set
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("⚠️  GOOGLE_API_KEY not set or still using placeholder")
        print("   Please edit .env file and add your real API key")
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Test with a simple request
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, respond with 'API connection successful'")
        
        if "successful" in response.text.lower():
            print("✅ Gemini API connection - OK")
            return True
        else:
            print("⚠️  Gemini API connection - Unexpected response")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API connection - Error: {e}")
        print("   Check your API key and internet connection")
        return False


def main():
    """Main setup function"""
    print("🚀 RAG-Anything with Gemini Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check LibreOffice
    libreoffice_ok = check_libreoffice()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Create directories
    create_directories()
    
    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
    
    # Test Gemini connection
    gemini_ok = test_gemini_connection()
    
    print("\n" + "=" * 40)
    print("📋 Setup Summary:")
    print(f"   Python: ✅")
    print(f"   Dependencies: ✅")
    print(f"   LibreOffice: {'✅' if libreoffice_ok else '⚠️'}")
    print(f"   Environment: ✅")
    print(f"   Gemini API: {'✅' if gemini_ok else '⚠️'}")
    
    if gemini_ok and libreoffice_ok:
        print("\n🎉 Setup complete! You can now use RAG-Anything with Gemini.")
        print("\n📚 Quick start:")
        print("   python examples/raganything_example.py document.pdf")
        print("   python examples/batch_raganything_example.py ./documents/")
    elif gemini_ok:
        print("\n✅ Setup mostly complete!")
        print("   You can process PDF and text files.")
        print("   Install LibreOffice to process DOCX/XLSX/PPTX files.")
    else:
        print("\n⚠️  Setup incomplete:")
        if not gemini_ok:
            print("   - Please add your GOOGLE_API_KEY to .env file")
        if not libreoffice_ok:
            print("   - Consider installing LibreOffice for full document support")
    
    return True


if __name__ == "__main__":
    main()
