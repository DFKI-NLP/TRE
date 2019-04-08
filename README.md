# Improving Relation Extraction by Pre-trained Language Representations

This repository contains the code of our paper:  
[Improving Relation Extraction by Pre-trained Language Representations.](https://openreview.net/forum?id=BJgrxbqp67) Christoph Alt*, Marc Hübner*, Leonhard Hennig

We fine-tune the pre-trained OpenAI GPT [1] to the task of relation extraction and show that it achieves state-of-the-art results on SemEval 2010 Task 8 and TACRED relation extraction datasets.

Our code depends on huggingface's PyTorch reimplementation of the OpenAI GPT [2] - so thanks to them.

## Installation

First, clone the repository to your machine and install the requirements with the following command:

```bash
pip install -r requirements.txt
```

We also need the weights of the pre-trained Transformer, which can be downloaded with the following command:
```
./download-model.sh
```

The english spacy model is required for sentence segmentation:
```
python -m spacy download en
```

## Prepare the data

We evaluate our model on [SemEval 2010 Task 8](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk) and [TACRED](https://catalog.ldc.upenn.edu/LDC2018T24), which is available through LDC.

Our model expects the input dataset to be in JSONL format. To convert a dataset run the following command:
```bash
python dataset_converter.py <DATASET DIR> <CONVERTED DATASET DIR> --dataset=<DATASET NAME>
```

## Training
E.g. for training on the TACRED dataset, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python relation_extraction.py train \
  --write-model True \
  --masking-mode grammar_and_ner \
  --batch-size 8 \
  --max-epochs 3 \
  --lm-coef 0.5 \
  --learning-rate 5.25e-5 \
  --learning-rate-warmup 0.002 \
  --clf-pdrop 0.1 \
  --attn-pdrop 0.1 \
  --word-pdrop 0.0 \
  --dataset tacred \
  --data-dir <CONVERTED DATASET DIR> \
  --seed=0 \
  --log-dir ./logs/
```

## Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python relation_extraction.py evaluate \
  --dataset tacred \
  --masking_mode grammar_and_ner \
  --test_file ./data/tacred/test.jsonl \
  --save_dir ./logs/ \
  --model_file <MODEL FILE (e.g. model_epoch...)> \
  --batch_size 8 \
  --log_dir ./logs/
```

## Citations
If you use our code in your research or find our repository useful, please consider citing our work.

```
@InProceedings{alt_improving_2019,
  author = {Alt, Christoph and H\"{u}bner, Marc and Hennig, Leonhard},
  title = {Improving Relation Extraction by Pre-trained Language Representations},
  booktitle = {Proceedings of AKBC 2019},
  year = {2019},
  url = {https://openreview.net/forum?id=BJgrxbqp67},
}
```

## License
lm-transformer-re is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## References
1. [Improving language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever.
2. [PyTorch implementation of OpenAI's Finetuned Transformer Language Model](https://github.com/huggingface/pytorch-openai-transformer-lm)
