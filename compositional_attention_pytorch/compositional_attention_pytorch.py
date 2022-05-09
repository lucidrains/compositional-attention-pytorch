import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from einops_exts import rearrange_many

def exists(val):
    return val is not None

class CompositionalAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        num_searches = 8,
        num_retrievals = 2,
        dropout = 0.,
        prenorm = False,
        causal = False
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()

        self.scale = dim_head ** -0.5
        inner_search_dim = dim_head * num_searches
        inner_retrieval_dim = dim_head * num_retrievals

        self.num_searches = num_searches
        self.num_retrievals = num_retrievals

        self.to_searches_queries = nn.Linear(dim, inner_search_dim, bias = False)
        self.to_searches_keys = nn.Linear(dim, inner_search_dim, bias = False)
        self.to_retrieval_values = nn.Linear(dim, inner_retrieval_dim, bias = False)

        self.to_retrieval_queries = nn.Linear(dim, inner_search_dim, bias = False)
        self.to_retrieval_keys = nn.Linear(dim_head, dim_head, bias = False)

        self.to_out = nn.Linear(inner_search_dim, dim, bias = False)

        self.search_dropout = nn.Dropout(dropout)
        self.retrieval_dropout = nn.Dropout(dropout)

        # autoregressive variant for self-experimentation
        self.causal = causal

    def forward(self, x, mask = None):
        """
        einstein notation:
        b - batch
        n - sequence dimension
        i - sequence dimension (source)
        j - sequence dimension (target, aggregation dimension)
        s - number of searches
        r - number of retrievals
        d - feature dimension
        """
        x = self.norm(x)

        s = self.num_searches
        r = self.num_retrievals

        # get search queries and keys

        sq, sk = self.to_searches_queries(x), self.to_searches_keys(x)
        sq, sk = rearrange_many((sq, sk), 'b n (s d) -> b s n d', s = s)

        sq = sq * self.scale

        # search similarity and attention

        search_sim = einsum('b s i d, b s j d -> b s i j', sq, sk)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            search_sim = search_sim.masked_fill(~mask, -torch.finfo(search_sim.dtype).max)

        if self.causal:
            i, j = search_sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
            search_sim = search_sim.masked_fill(causal_mask, -torch.finfo(search_sim.dtype).max)

        search_sim = search_sim - search_sim.amax(dim = -1, keepdim = True).detach()
        search_attn = search_sim.softmax(dim = -1)
        search_attn = self.search_dropout(search_attn)

        # get retrieval values

        rv = self.to_retrieval_values(x)
        rv = rearrange(rv, 'b n (r d) -> b r n d', r = r)

        retrieved = einsum('b s i j, b r j d -> b s r i d', search_attn, rv)

        # get retrieval queries and keys

        rq, rk = self.to_retrieval_queries(x), self.to_retrieval_keys(retrieved)
        rq = rearrange(rq, 'b n (s d) -> b s n d', s = s)
        rq = rq * self.scale

        # get retrieval attention

        retrieval_sim = einsum('b s n d , b s r n d -> b s n r', rq, rk)

        retrieval_sim = retrieval_sim - retrieval_sim.amax(dim = -1, keepdim = True).detach()
        retrieval_attn = retrieval_sim.softmax(dim = -1)
        retrieval_attn = self.retrieval_dropout(retrieval_attn)

        out = einsum('b s n r, b s r n d -> b s n d', retrieval_attn, retrieved)
        out = rearrange(out, 'b s n d -> b n (s d)')

        return self.to_out(out)
