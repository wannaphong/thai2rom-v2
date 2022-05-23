# Roman-Thai Transformer Models

Model (4 epoch): https://huggingface.co/wannaphong/Roman2Thai-transliterator

Training report: https://wandb.ai/wannaphong/en-th-transformer/reports/Roman-Thai-Transformer-Models--VmlldzoyMDU0MTIx?accessToken=69cefyoo260297cg7x54jr70czj0zow03jbu45i3js3y2ide38lbcn74pnnbfjzx

- Use pretrained model from [Helsinki-NLP/opus-mt-en-mul](https://huggingface.co/Helsinki-NLP/opus-mt-en-mul).
- English-> Multiple languages (and Thai)
- model: transformer
- pre-processing: normalization + SentencePiece (spm32k,spm32k)

**CER**

from testset

| Epochs | CER    |
| ------ | ------ |
| 1      | 17.52% |
| 3      | 15.06% |
| 4      | 14.48% |

