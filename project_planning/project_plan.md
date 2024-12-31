# BrahmaSumm Project Plan

## Project Overview
BrahmaSumm is an advanced document summarization and visualization tool that focuses on efficient document processing and analysis. The tool uses cutting-edge chunking and clustering techniques to reduce token usage sent to LLMs by up to 99% while maintaining content quality. It provides support for multiple document formats and creates intuitive visualizations for better document understanding.

## Key Features
1. **Multi-format Support**
   - PDFs, YouTube videos, audio files, HTML
   - Spreadsheets and Google Drive folders
   - Automatic format detection and processing

2. **Smart Document Processing**
   - Intelligent chunking with structure preservation
   - Clustering-based summarization
   - Token usage optimization (up to 99% reduction)

3. **Visualization Capabilities**
   - UMAP-based cluster visualization
   - Document flow visualization
   - Theme-based cluster representation

4. **Data Extraction**
   - Table extraction
   - Image processing
   - Text extraction and vectorization

5. **Integration Features**
   - Multiple LLM provider support (Groq, Ollama, OpenAI)
   - Embedding model flexibility
   - API integration capabilities

## Main Scripts Analysis

### src/summarize.py
The main entry point for document processing that orchestrates the entire summarization pipeline. It handles document loading, chunking, clustering, and theme extraction using the configured LLM provider. The script generates both textual summaries and visual representations of the document structure.

### src/doc_loaders/multimedia_loader.py
Handles the loading and processing of multimedia content, particularly focusing on audio and YouTube video transcription. It uses faster-whisper for transcription and includes robust error handling for ffmpeg dependencies. The script supports both direct audio file processing and YouTube video audio extraction.

### src/visualize/visualize.py
Provides visualization capabilities using UMAP for dimensionality reduction and cluster visualization. It creates intuitive visual representations of document clusters and their relationships. The script supports both grid-based label visualization and scatter plot-based cluster visualization with theme labeling.

### src/models/models.py
Manages different LLM and embedding model providers (Groq, Ollama, OpenAI) with configuration-based model loading. It handles token counting, model initialization, and provides a unified interface for model interactions. The script includes error handling for missing API keys and configuration issues.
