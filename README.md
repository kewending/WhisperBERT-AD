# WhisperBERT-AD  
> Multimodal Speech + Text Deep Learning Framework for Alzheimer's Disease Detection  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg) ![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)  

## 📖 Overview  
WhisperBERT-AD is an **end-to-end multimodal deep learning framework** for detecting Alzheimer’s Disease (AD) from spontaneous speech.  
It combines **OpenAI’s Whisper** (for high-fidelity acoustic embeddings) and **BERT** (for linguistic embeddings) using a **cross-attentive fusion mechanism**.  
This design enables rich **intra- and inter-modal interactions**

Key features include:  
- 🗣 **Whisper as an acoustic encoder** – direct audio feature extraction, eliminating log-Mel spectrogram preprocessing.  
- 📝 **BERT as a text encoder** – extracting contextual linguistic representations.  
- 🔀 **Cross-attentive fusion** – fine-grained multimodal representation learning between Whisper and BERT embeddings.  
- ⚡ **Optimal Transport Kernel Embedding (OTKE)** – compresses variable-length Whisper embeddings into fixed-size vectors, harmonizing with BERT.
---

## 🚀 Installation  

### Requirements  
- Python 3.9+  
- [PyTorch](https://pytorch.org/) with CUDA (if GPU available)  
- Hugging Face Transformers  
- Datasets + Evaluation libraries  

Install dependencies:  
```bash
git clone https://github.com/kewending/WhisperBERT-AD.git
cd WhisperBERT-AD
pip install -r requirements.txt
```

## 📂 Repository Structure
```bash
WhisperBERT-AD/
│── configs/          # Experiments configs
│── data/             # ADReSS dataset etc
│── models/           # Saved model outputs
│── results/          # Experimental results 
│── src/              # Model Source code
│── requirements.txt
│── README.md
│── run.py                    
```
## 📊 Dataset
This project uses the ADReSS Challenge dataset:

The dataset should be organized in the following structure:
```
data/
├── ADReSS
    ├── Train/
    │   ├── audio1.wav
    │   ├── ...
    │   └── metadata.csv (with id, label, ..)
    └── Test/
        ├── audio1.wav
        ├── ...
        └── metadata.csv
```

## ⚡Usage
### Training
```bash
python run.py --config "configs/VAS_AudioClassifier.yaml"
```
## ⭐ Acknowledgements
- [Optimal Transport Kernel Embedding](https://github.com/claying/OTK)
- [Hugging Face](https://huggingface.co/docs/transformers/index)

## 📜 Citation
If you use this work, please cite:
```bibtex
@misc{ding2025whisperbertad,
  author       = {Kewen Ding},
  title        = {WhisperBERT-AD: Cross-Attentive Multimodal Fusion of Whisper and BERT for Alzheimer’s Disease Detection},
  year         = {2025},
  url          = {https://github.com/kewending/WhisperBERT-AD}
}
```