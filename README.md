<img src="./compositional-attention.png" width="400px"></img>

## Compositional Attention - Pytorch

Implementation of <a href="https://arxiv.org/abs/2110.09419">Compositional Attention</a> from MILA. They reframe the "heads" of multi-head attention as "searches", and once the multi-headed/searched values are aggregated, there is an extra retrieval step (using attention) off the searched results. They then show this variant of attention yield better OOD results on a toy task. Their ESBN results still leaves a lot to be desired, but I like the general direction of the paper.

## Install

```bash
$ pip install compositional-attention-pytorch
```

## Usage

```python
import torch
from compositional_attention_pytorch import CompositionalAttention

attn = CompositionalAttention(
    dim = 1024,            # input dimension
    dim_head = 64,         # dimension per attention 'head' - head is now either search or retrieval
    num_searches = 8,      # number of searches
    num_retrievals = 2,    # number of retrievals
    dropout = 0.,          # dropout of attention of search and retrieval
)

tokens = torch.randn(1, 512, 1024)  # tokens
mask = torch.ones((1, 512)).bool()  # mask

out = attn(tokens, mask = mask) # (1, 512, 1024)
```

## Citations

```bibtex
@article{Mittal2021CompositionalAD,
    title   = {Compositional Attention: Disentangling Search and Retrieval},
    author  = {Sarthak Mittal and Sharath Chandra Raparthy and Irina Rish and Yoshua Bengio and Guillaume Lajoie},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2110.09419}
}
```
