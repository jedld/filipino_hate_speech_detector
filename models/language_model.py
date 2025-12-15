import torch
import torch.nn as nn
from torch import Tensor


class MiniTransformerLanguageModel(nn.Module):
    """A lightweight autoregressive Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_position_embeddings = max_position_embeddings
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_multiplier = ff_multiplier

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_position_embeddings, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.lm_head.weight)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute logits over the vocabulary for each position."""
        hidden = self.encode(input_ids)
        logits = self.lm_head(hidden)
        return logits

    def encode(self, input_ids: Tensor, causal_mask: bool = True) -> Tensor:
        """Return the contextualized hidden states prior to the LM head."""
        if input_ids.size(1) > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds max positional embeddings {self.max_position_embeddings}"
            )

        positions = self.pos_embedding[:, : input_ids.size(1), :]
        hidden_states = self.token_embedding(input_ids) + positions
        hidden_states = self.dropout(hidden_states)

        attention_mask = None
        if causal_mask:
            attention_mask = self._generate_square_subsequent_mask(input_ids.size(1), input_ids.device)
        transformed = self.transformer(hidden_states, mask=attention_mask)
        transformed = self.layer_norm(transformed)
        return transformed

    @staticmethod
    def _generate_square_subsequent_mask(seq_len: int, device: torch.device) -> Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> Tensor:
        """Autoregressively generate tokens continuing the provided prompt."""
        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.max_position_embeddings :]
            logits = self(context)
            next_token_logits = logits[:, -1, :]

            if temperature <= 0:
                raise ValueError("temperature must be positive")
            next_token_logits = next_token_logits / temperature

            if top_k > 0:
                values, _ = torch.topk(next_token_logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_values,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tokens], dim=1)
        return generated