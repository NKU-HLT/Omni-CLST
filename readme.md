# Omni-CLST: Error-Aware Curriculum Learning with Guided Selective Chain-of-Thought for Audio Question Answering  

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Omni--CLST-blue.svg)](https://huggingface.co/Kiri233/Omni-CLST)  [![arXiv](https://img.shields.io/badge/arXiv-2501.12345-b31b1b.svg)](https://arxiv.org/abs/2509.12275)


With the rapid progress of large audio-language models (LALMs), audio question answering (AQA) has emerged as a challenging task requiring both fine-grained audio understanding and complex reasoning. While current methods mainly rely on constructing new datasets via captioning or reasoning traces, existing high-quality AQA data remains underutilized. To address this, we propose Omni-CLST, an error-aware Curriculum Learning framework with guided Selective Chain-of-Thought. The framework efficiently leverages existing high-quality dataset through two key strategies: an error-aware curriculum that organizes samples by difficulty, and a guided thought dropout mechanism that focuses reasoning on challenging cases. Experiments show that Omni-CLST achieves 73.80% on MMAU-mini and a new state of the art of 64.30% on MMAR, demonstrating robust generalization in multimodal audio-language understanding.

<div align="center">
  <img src="asset/model.png" alt="Workflow" width="750"/>
</div>

## 📀 Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/NKU-HLT/Omni-CLST.git
    cd Omni-CLST
    ```

2. Install the required dependencies:

    ```bash
    conda create -n omniclst python=3.10 -y
    conda activate omniclst
    pip install -r requirements.txt
    ```

## ⭐ Training

1. **Prepare the Dataset**: Download and preprocess the [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/), [Audsemthinker-mc_qa](https://huggingface.co/datasets/gijs/audsem/tree/main) and [our jsonl file](https://huggingface.co/Kiri233/Omni-CLST/tree/main/data).

Put our jsonl file in the `data` folder. The structure should look like this:

    ```
    Omni-CLST
    ├── data
    │   ├── sft_guided_drop_thought
    │   │   ├── train.jsonl
    │   │   ├── valid.jsonl
    │   └── CLST_15k.jsonl
    ```

2. **SFT**: Use the following command to start sft:

    ```bash
    cd script
    bash sft_lora.sh
    ``` 
3. **GRPO**: Use the following command to start grpo:

    ```bash
    cd script
    # will automatically inference the benchmark for the latest checkpoint
    bash CLST.sh 
    ```

## ⭐ Inference

1. **Prepare the Dataset**: Download and preprocess the [MMAU-mini](https://github.com/Sakshi113/mmau) and [MMAR](https://github.com/ddlBoJack/MMAR)

2. **Inference**: Use the following command to start inference:

    ```bash
    cd script
    bash infer.sh "checkpoint_path"
    ```


## 🤝🏻 Contact
Should you have any questions, please contact zhaojinghua_hlt@mail.nankai.edu.cn

## 📚 Citation
Coming soon.

## 🙏 Acknowledgment:
This code is based on the [Ke-Omni-R](https://github.com/shuaijiang/Ke-Omni-R/) repositories. 