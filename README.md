# Audio Source Separation Using Semi-Supervised Learning

## Abstract

Audio source separation is a fundamental task in audio signal processing, aimed at isolating individual sound sources (such as vocals, drums, bass, etc.) from a mixture. Traditional approaches often rely heavily on supervised learning, requiring large amounts of annotated data, which is often expensive and labor-intensive to obtain. This project explores a **semi-supervised learning** approach for audio source separation using **Generative Adversarial Networks (GANs)**. Inspired by techniques in the paper *Semi-supervised Monaural Singing Voice Separation with a Masking Network Trained on Synthetic Mixtures* ([arXiv:1812.06087](https://arxiv.org/abs/1812.06087)), this implementation expands the idea to **stereo audio** and **multiple audio sources** beyond vocals, including drums, bass, and other instrumental components.

By leveraging both labeled and unlabeled data, the model aims to learn robust source separation capabilities while mitigating the need for extensive annotated datasets. The use of adversarial training allows the system to generalize better and produce higher quality audio separations even from challenging stereo mixtures.

---

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
  - [1. Setting up the Environment](#1-setting-up-the-environment)
  - [2. Downloading the Dataset & Preprocessing](#2-downloading-the-dataset--preprocessing)
  - [3. Training](#3-training)
  - [4. Testing](#4-testing)
  - [5. Pretrained Model & Evaluation](#5-pretrained-model--evaluation)
- [Contribution](#contribution)

---

## Introduction

Audio source separation remains a critical challenge in the field of music information retrieval and audio signal processing. It has practical applications in music remixing, karaoke systems, audio enhancement, and preprocessing for downstream audio tasks. While significant progress has been made using supervised deep learning techniques, these approaches often suffer from the scarcity of clean and diverse training data.

This project is a research-driven implementation that builds upon the semi-supervised learning architecture introduced in the paper *Semi-supervised Monaural Singing Voice Separation with a Masking Network Trained on Synthetic Mixtures* ([arXiv:1812.06087](https://arxiv.org/abs/1812.06087)). However, instead of focusing solely on **monaural vocals**, this project extends the architecture to handle **stereo inputs** and separates **multiple sources**, including **vocals**, **drums**, **bass**, and **others**.

Our implementation is designed to explore and evaluate how generative models, particularly GANs, can be used to improve the quality and accuracy of source separation in under-labeled or mixed-labeled data conditions. The goal is to reduce dependency on large supervised datasets and push the boundaries of what semi-supervised architectures can achieve in multi-source, stereo environments.

---

## Requirements

To run this project, the following requirements must be met:

- Python 3.10
- All required packages are listed in the `environment.yaml` file.

---

## How to Use

### 1. Setting up the Environment

You can choose one of two methods:

**Using pip and `requirements.txt`:**
```bash
pip install -r requirements.txt
```

**Using Anaconda and `environment.yaml`:**
```bash
conda env create -f environment.yaml
conda activate <your-environment-name>
```

### 2. Downloading the Dataset & Preprocessing

**Datasets:**
- [DSD100 Dataset](https://sigsep.github.io/datasets/dsd100.html)
- [MUSDB18 Dataset](https://sigsep.github.io/datasets/musdb.html)

**Creating Spectrograms:**
To preprocess the dataset and generate spectrograms for training, use the following commands:

**For DSD100:**
```bash
python dataset.py --dataset DSD100 --vocals
python dataset.py --dataset DSD100 --bass
python dataset.py --dataset DSD100 --drums
```

**For MUSDB18:**
```bash
python dataset.py --dataset MUSDB18 --vocals
python dataset.py --dataset MUSDB18 --bass
python dataset.py --dataset MUSDB18 --drums
```

This will create the required spectrogram files for training.

### 3. Training

Before training, edit the `test.py` file's main guard to set the correct `--target` (vocals, bass, drums) as per your needs.

To begin training, use:
```bash
python train.py --config configs/vocals_new.yaml
```

You can configure checkpoint saving, evaluation intervals, target sources, and other training parameters in the `configs/` directory.


### 4. Pretrained Model & Evaluation

To separate music channels (default: vocals and accompaniment) using the pretrained model:
```bash
python create_songs.py
```

To run the same on your own custom audio files:

1. Place your audio files in a folder named `./input/`
2. Then run:
```bash
python create_songs.py --input custom
```

The separated outputs will be saved in the `./outputs/` directory.

---

## Contribution

**This section provides instructions and details on how to submit a contribution via a pull request. It is important to follow these guidelines to make sure your pull request is accepted.**
1. Before choosing to propose changes to this project, it is advisable to go through the readme.md file of the project to get the philosophy and the motive that went behind this project. The pull request should align with the philosophy and the motive of the original poster of this project.
2. To add your changes, make sure that the programming language in which you are proposing the changes should be the same as the programming language that has been used in the project. The versions of the programming language and the libraries(if any) used should also match with the original code.
3. Write a documentation on the changes that you are proposing. The documentation should include the problems you have noticed in the code(if any), the changes you would like to propose, the reason for these changes, and sample test cases. Remember that the topics in the documentation are strictly not limited to the topics aforementioned, but are just an inclusion.
4. Submit a pull request via [Git etiquettes](https://gist.github.com/mikepea/863f63d6e37281e329f8) 

---

## License

Please refer to the project's license file for usage terms and conditions.

## Citation
This project is inspired ffrom the following paper
```bibtex
  @inproceedings{michelashvili2018singing,
  title={Semi-Supervised Monaural Singing Voice Separation With a Masking Network Trained on Synthetic Mixtures},
  author={Michael Michelashvili and Sagie Benaim and Lior Wolf},
  booktitle={ICASSP},
  year={2019},
}
```
