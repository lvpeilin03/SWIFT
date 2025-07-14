# SWIFT: Self-Reflective Knowledge Switching for Misinformation Detection
This includes the original implementation of SWIFT: Self-Reflective Knowledge Switching for Misinformation Detection by Peilin Lv.

SWIFT is a novel framework (Figure bellow) that enables LLMs to dynamically assess their confidence in their internal knowledge and decide whether to rely on their own understanding or retrieve external evidence. SWIFT operates by adjusting non-knowledge-center nodes within the model to enable adaptive switching without the need for extensive retraining. Our approach closes the temporal gap in misinformation detection while significantly reducing computational overhead. 

We evaluated SWIFT on five challenging benchmarks and demonstrated that it outperforms existing adaptive strategies in terms of both accuracy and efficiency. SWIFT offers a scalable and effective solution for real-time misinformation detection, thereby advancing the capabilities of LLMs in dynamic information environments. 

<img width="9130" height="4950" alt="Image" src="https://github.com/user-attachments/assets/40523fe4-86cd-45e4-951a-16188a059ec0" />

If you find our code, data, models, or the paper useful, please cite the paper:

## Content
1. Installation
2. Retriever Setup
3. Datasets
4. Training
5. Evaluation

## Installation
Install dependent Python libraries by running the command below.
`pip install -r requirements.txt`
Please use the latest version of vllm, as the older version may not enable you to set skip_special_tokens via SamplingParam, which is added by [this PR](https://github.com/vllm-project/vllm/issues/893).

## Retriever Setup

By default, we use [Contriever](https://github.com/facebookresearch/contriever) as our retrieval component.

### Download data

Download preprocessed passage data used in DPR.
`cd retrieval_lm
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz`

Then, download the generated passages. We use [Contriever-MSMARCO](https://huggingface.co/facebook/contriever-msmarco)
`wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar`
### Run retriever

You can run passage retrieval by running the command below.
`cd retrieval_lm
python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco --passages psgs_w100.tsv \
    --passages_embeddings "wikipedia_embeddings/*" \
    --data YOUR_INPUT_FILE  \
    --output_dir YOUR_OUTPUT_FILE \
    --n_docs 20`
Your input file should be either a json or jsonl. Each instance must contain either question or instruction, which will be used as a query during retrieval.

## Datasets
You can download all datasets at [huggingface](https://huggingface.co).

## Training
(1) Get activations by running `python generation_sim.py liar`. Here we take the Liar dataset as an example, and head-wise activations are stored in the features folder(You can also get layer-wise activations by adjust code if you need).
(2) Train logistic classifiers for each attention head by running `python get_classify.py liar 12`, the numbers that follow represent the top 12 classifiers that get the best performance.
(3) Selection of the best performing team of experts by running `python get_combination.py`
(4) Run `python test_lora.py` to fine-tune model and get training data.
(5) Then, you can run passage retrieval by running the command below.
`cd retrieval_lm
python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco --passages psgs_w100.tsv \
    --passages_embeddings "wikipedia_embeddings/*" \
    --data YOUR_INPUT_FILE  \
    --output_dir YOUR_OUTPUT_FILE \
    --n_docs 20`
Your input file should be either a json or jsonl. Each instance must contain either question or instruction, which will be used as a query during retrieval.
(6) Retrain expert team by running
`python get_classify_qav.py liar 12
python get_combination_qav.py liar
python last_qav.py liar`

## Evaluation
(1) Run `python evaluation.py liar` to get first part.
(2) Run this code to get retrieved messages.
`cd retrieval_lm
python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco --passages psgs_w100.tsv \
    --passages_embeddings "wikipedia_embeddings/*" \
    --data YOUR_INPUT_FILE  \
    --output_dir YOUR_OUTPUT_FILE \
    --n_docs 20`
(3) Run this code to get remaining part.
`python evaluation2.py liar`


