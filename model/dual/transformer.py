import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat

# ============================================================
# ============================================================

class ResidualWrapper(Module):
    def __init__(self, branch):
        super().__init__()
        self.branch = branch

    def forward(self, x):
        result = self.branch(x)
        if isinstance(result, tuple):
            return (x + result[0],) + result[1:]
        return x + result


class SimpleEmbed(Module):
    def __init__(self, dim, num_discrete=(), num_continuous=0):
        super().__init__()
        self.categ_embeddings = ModuleList([
            nn.Embedding(num_cat, dim) for num_cat in num_discrete
        ])
        self.num_continuous = num_continuous
        if num_continuous > 0:
            self.numerical_weights = nn.Parameter(torch.randn(num_continuous, dim))
            self.numerical_biases = nn.Parameter(torch.randn(num_continuous, dim))

    def forward(self, inputs):
        x_categ, x_numer = inputs
        tokens = []
        for i, embed in enumerate(self.categ_embeddings):
            tokens.append(embed(x_categ[:, i]).unsqueeze(1))
        if self.num_continuous > 0:
            x_num = x_numer.unsqueeze(-1)
            num_tokens = x_num * self.numerical_weights + self.numerical_biases  # (b, C_num, 1) * (C_num, dim) + (C_num, dim) → (b, C_num, dim)
            tokens.append(num_tokens)
        return torch.cat(tokens, dim=1)


class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn


class Transformer(Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        num_residual_streams = 4
    ):
        super().__init__()

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                ResidualWrapper(branch = Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                ResidualWrapper(branch = FeedForward(dim, dropout = ff_dropout)),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            x, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = ff(x)

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)


class NumericalEmbedder(Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases


class FTTransformer(Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_residual_streams = 4
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'


        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)


        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens


        categories_with_special = tuple(c + num_special_tokens for c in categories)

        self.embedding = SimpleEmbed(dim, num_discrete = categories_with_special, num_continuous = num_continuous)

        # ---- CLS token ----

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))


        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            num_residual_streams = num_residual_streams
        )


        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False, return_features = False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        x_categ = x_categ + self.num_special_tokens

        x = self.embedding((x_categ, x_numer))


        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)


        x, attns = self.transformer(x, return_attn = True)
        # x: (b, 1+C_cat+C_num, dim), attns: (depth, b, heads, seq_len, seq_len)


        x = x[:, 0]


        if return_features:
            return x


        logits = self.to_logits(x)              # (b, dim) → LayerNorm → ReLU → Linear → (b, dim_out)

        if not return_attn:
            return logits

        return logits, attns
