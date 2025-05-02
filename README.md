# DATA 266 - Lab 2: Advanced AI Applications

This repository contains implementations for the three parts of the DATA 266 Lab 2 assignment, covering MultiModal RAG systems, fine-tuning Stable Diffusion, and developing an agentic AI Travel Assistant.

## Table of Contents

- [Part 1: MultiModal Retrieval-Augmented Generation](#part-1-multimodal-retrieval-augmented-generation)
- [Part 2: Fine-Tuning Stable Diffusion for Image Generation](#part-2-fine-tuning-stable-diffusion-for-image-generation)
- [Part 3: Developing an Agentic AI Travel Assistant](#part-3-developing-an-agentic-ai-travel-assistant)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Part 1: MultiModal Retrieval-Augmented Generation

A MultiModal RAG system that retrieves relevant text and images from a document to generate accurate responses to economic queries.

### Features

- **Data Processing & Storage**: Extracts text and image embeddings from a PDF document and stores them in ChromaDB.
- **Retrieval with Indexing & Ranking**: Implements efficient indexing and retrieval based on cosine similarity.
- **Response Generation**: Uses GPT-4o-mini to generate responses based on retrieved content.
- **Content Extraction**: Precisely extracts figures and tables with captions from PDF documents.

### Implementation Details

- Uses sentence-transformers (E5-large) for text embeddings
- CLIP embeddings for image matching
- PDF parsing with PyMuPDF and pdfplumber
- ChromaDB for vector storage and retrieval
- Contextual ranking and hybrid search for improved retrieval accuracy

## Part 2: Fine-Tuning Stable Diffusion for Image Generation

Fine-tuning of Stable Diffusion on a landscape dataset to create high-quality, diverse landscape images.

### Features

- **Dataset Preparation**: Custom landscape dataset with preprocessing to 512x512 resolution.
- **Fine-Tuning**: Implementation of Stable Diffusion fine-tuning with optimization techniques.
- **Image Generation**: Generation of high-quality landscape images from text prompts.
- **Evaluation**: Assessment using Inception Score (IS) and CLIP Similarity Score.

### Implementation Details

- Uses Diffusers and Transformers libraries for model handling
- AdamW optimizer with cosine scheduler with restarts
- Data augmentation including horizontal flips, rotations, and color adjustments
- Post-processing to enhance image quality
- Comprehensive evaluation metrics visualization

## Part 3: Developing an Agentic AI Travel Assistant

An AI travel assistant with autonomous agents that interact with external APIs to provide comprehensive travel planning.

### Features

- **Flight API Agent**: Retrieves flight options based on destination and dates.
- **Weather API Agent**: Fetches weather forecasts for the travel destination.
- **Hotel API Agent**: Finds accommodation options categorized by price range.
- **Itinerary Planner Agent**: Synthesizes information to generate a personalized travel itinerary.

### Implementation Details

- Uses CrewAI framework for agent orchestration
- Integrates with Amadeus API for flight data
- OpenWeatherMap API for weather forecasts
- Hotelbeds API for hotel information
- Google's Gemini 1.5 Pro for natural language generation
- Error handling and fallback mechanisms
- Token caching and rate limiting to optimize API usage

## Getting Started

These instructions will help you set up and run the projects on your local machine.

### Prerequisites

- Python 3.8+
- Pip package manager
- GPU (recommended for Part 2)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data266-lab2.git
cd data266-lab2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages for each part:
```bash
pip install -r requirements.txt
```

### Part-specific setup

#### Part 1: MultiModal RAG

```bash
pip install PyMuPDF pdfplumber pandas sentence_transformers chromadb openai
```

#### Part 2: Stable Diffusion Fine-Tuning

```bash
pip install diffusers transformers accelerate datasets ftfy bitsandbytes xformers safetensors huggingface_hub torchvision scipy torchmetrics lpips open_clip_torch pytorch-fid
```

#### Part 3: Agentic Travel Assistant

```bash
pip install requests python-dotenv tenacity crewai langchain langchain-google-genai
```

## Usage

### Part 1: MultiModal RAG

1. Place the document PDF in the project directory
2. Run the RAG system:
```bash
python Lab2_Part1.py
```
3. The system will process the document, extract content, and create the necessary embeddings
4. Results will be saved to `optimized_submission.csv`

### Part 2: Stable Diffusion Fine-Tuning

1. Prepare your dataset in the specified directory
2. Run the fine-tuning script:
```bash
python lab2_part2.py
```
3. Generated images will be saved to the specified output directory
4. Evaluation metrics will be displayed and saved

### Part 3: Agentic Travel Assistant

1. Create an `api.env` file with the required API keys:
```
AMADEUS_API_KEY=your_key
AMADEUS_API_SECRET=your_secret
OPENWEATHERMAP_API_KEY=your_key
HOTELBEDS_API_KEY=your_key
HOTELBEDS_SECRET=your_secret
GOOGLE_API_KEY=your_key
```

2. Run the travel assistant:
```bash
python Lab2_Part3.py
```

3. Follow the prompts to input travel details (origin, destination, dates)
4. The system will generate a comprehensive travel itinerary

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to Kaggle for hosting the competition for Part 1
- HuggingFace for providing pretrained models and diffusers library
- External API providers for enabling the agentic travel assistant
