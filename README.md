# LLaVA on PMC-VQA

This repository contains preprocessing, inference, and evaluation scripts for running LLaVA v1.5 on the **PMC-VQA** medical visual question answering dataset. It covers:

- Converting the official PMC-VQA CSV files into LLaVA style JSON with open ended question Answering 
- Running a zero shot LLaVA baseline
- Running inference with fine tuned LLaVA checkpoints
- Evaluating with token level EM and F1
- Evaluating with an external LLM judge via Ollama

PMC-VQA is a large scale MedVQA dataset with about 227k QA pairs over 149k images that span multiple imaging modalities and diseases.

## Files Included 

├── baseline.py # baseline training and inference code
├── preprocess.py # script to preprocess raw PMC-VQA data
├── split_dataset.py # script to split dataset into train, val, test
├── epoch1_inference.py # inference logic after first epoch
├── eval_accuracy_token.py # token-level evaluation script
├── gpt_eval.py # LLM-based evaluation script (optional)
├── *.json # processed datasets, predictions, eval outputs
├── *.txt / *.log / *.png # logs, training loss curves, evaluation results
├── .gitignore
├── .gitattributes # version control configuration
└── README.md # project documentation

## 1. Environment and Dependencies

This project assumes you already have a working LLaVA v1.5 environment. Ensure to download the model links from the Google Drive links 

### Core dependencies

- Python 3.9 or later
- PyTorch with CUDA (recommended)
- `llava` library and its dependencies (from the official LLaVA repo)
- `pandas`
- `scikit-learn`
- `tqdm`
- `requests`
- `Pillow`

Example setup (adapt to your CUDA version and preferences):

```bash
conda create -n llava-pmc-vqa python=3.12 -y
conda activate llava-pmc-vqa

# Install PyTorch (pick the command appropriate for your system)
pip install "torch>=2.0" "torchvision"

# Install LLaVA and its dependencies
git clone --recurse-submodules https://github.com/devangguptaa/llava-pmc-vqa.git
cd llava-pmc-vqa 
cd LLaVA
pip install -e .

cd .. 
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/RadGenome/PMC-VQA 
cd PMC-VQA
git pull --include==images.zip 
cd .. 

# Back to this repo
pip install pandas scikit-learn tqdm requests pillow

