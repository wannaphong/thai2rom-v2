# Roman-Thai Seq2seq with attention

Training report: https://wandb.ai/wannaphong/en-th-thai_romanize_pytorch_seq2seq_attention/reports/Roman-Thai-Seq2seq-with-attention--VmlldzoyMDU0Mjc0?accessToken=knnux4go3k6xi1fc1r98c409u73xym3ul7i0kuy5sjukzl0e50fr73nbyf2eq7lg


- Thank you Lalita Lowphansirikul for code. https://github.com/PyThaiNLP/pythainlp/pull/246 and https://github.com/PyThaiNLP/pythainlp/issues/202
- Train Roman-Thai Seq2seq with attention
- Max 25 epoch
- Use 15 epoch for training Thai-English transliterate 

**CER**

from testset

| Epochs | CER    |
| ------ | ------ |
| 14     | 34.75% |
| 15     | 34.79% |
| 25     | 32.92% |

