# Word Embedding Learning Network for Small Data Applications in NLP

### Paper 
This code accompanies the paper titled "Domain Lexical Knowledge-Based Word Embedding Learning for Text Classification Under Small Data," which has been submitted to Data-Centric Engineering. A citation link will be provided upon acceptance.

This repository contains the implementation of a transformer-based language model (LM) for word embedding learning, aimed at improving classification performance, particularly in applications with small datasets.


## Applications
We provide three example applications:
- **Emotion Recognition**
- **Sentiment Analysis**
- **Question Answering**

Datasets, knowledge bases, and corresponding learning network checkpoint files are available in the following directories:
- `/data/dataset/`: Contains datasets for the applications.
- `/data/knowledgebase/`: Stores the knowledge bases.
- `/mappingmodel/`: Includes the learning network checkpoint files.

## Custom Processing
If you want to run custom processes, the code is available in `/code/basemodel_path/*kefPL.py`. You can choose from the following base models:
- `Bi-LSTM att`
- `DualCL`
- `Kil`
- `LCL`

### Knowledge Base Collection
To collect a knowledge base for your own downstream tasks, run the following scripts in `/code/knowledgecollection/` in sequence:
1. `extreackWords.py`
2. `augKnowledge.py`
3. `select_unique_token.py`

## Experiment Preparation
### Dependencies
Ensure the following dependencies are installed:
- `python>=3.6`
- `torch>=1.7.1`
- `datasets>=1.12.1`
- `transformers>=4.9.2` (Hugging Face)

Alternatively, install all required dependencies using:
```bash
pip install -r requirements.txt


