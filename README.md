# tai-scan-rag-official

RAG (Retrieval-Augmented Generation) component of the TAI Scan Tool.

## Overview

This project provides a RAG pipeline for analyzing AI systems against the EU AI Act. It retrieves relevant legal documents (recitals, articles, annexes) and generates compliance assessments using configurable LLM backends.

## Features

- **Document Retrieval**: Embeds and searches EU AI Act documents using Annoy for fast approximate nearest neighbor search
- **Multiple Embedding Backends**: HuggingFace, Ollama, or Fraunhofer FHG Gateway
- **Multiple Text Generation Backends**: HuggingFace, Ollama, or Fraunhofer FHG Gateway
- **Task Templates**: Pre-configured prompts for risk classification, relevant resource identification, and obligation generation

## Architecture

```
NaiveRAG (BaseRAG)
├── Embedding Generator (HuggingFace / Ollama / FHG Gateway)
├── Annoy Index (vector similarity search)
└── Text Generator (HuggingFace / Ollama / FHG Gateway)
```

## Configuration

Configuration is managed via YAML files. Key settings:

- `Experiment`: cache directories, model names, dataset paths, embeddings storage
- `RAG`: embedding framework, text generation framework, k-best retrieval parameters
- `front-end`: role, domain, type, input data, intended use (for query templates)

## Environment Variables

Required for Docker deployment:

| Variable | Description |
|----------|---------------------------|
| `FHG_GATEWAY_BASE_URL` | Fraunhofer Gateway API base URL |
| `RAG_AUTH_TOKEN` | Authentication token for the Fraunhofer Gateway |

## Usage

```bash
python run.py <config_path>
```

The pipeline:
1. Loads configuration and EU AI Act dataset
2. Generates embeddings for documents (or loads cached embeddings)
3. Embeds the user query
4. Retrieves relevant documents via Annoy similarity search
5. Generates responses using the configured LLM backend

## Project Structure

```
├── RAGs/                    # Core RAG components
│   ├── base_conf.py         # Base configuration class
│   ├── base_RAG.py          # Abstract RAG base class
│   ├── naive_RAG.py         # NaiveRAG implementation
│   ├── inference_handler.py # Embedding & text generation backends
│   ├── llm_confs.py         # LLM configuration classes
│   └── utils.py             # Utility functions
├── configs/                 # YAML configuration files
├── datasets/                # EU AI Act document data
├── mkdata/                  # Data processing scripts
├── run.py                   # Main entry point
└── README.md
```

## Dependencies

- Python 3.10+
- torch, numpy, transformers
- openai (for FHG Gateway)
- ollama
- annoy
- scikit-learn
- matplotlib
- pyyaml
