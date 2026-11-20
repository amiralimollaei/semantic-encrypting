"""Model probabilities for plaintext compression and reversible cover text."""

from collections.abc import Callable
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import Cache

from .arithmetic import MAX_TOTAL, ArithmeticDecoder, ArithmeticEncoder, PrefixInterval

_FREQUENCY_BUDGET = 1 << 22
_COVER_CHOICES = 32


class RenderError(ValueError):
    """The ciphertext could not be represented as reversible text."""


class TextCodec:
    """Use the full vocabulary for plaintext and canonical tokens for covers."""

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
        ))
        # Transformers' decorated .to annotation retains an extra self parameter.
        move_model = cast(Callable[[torch.device], PreTrainedModel], model.to)
        move_model(torch.device(device))
        self._model = model.eval()
        bos = self._model.config.bos_token_id
        if bos is None:
            raise ValueError("Model must define a beginning-of-sequence token")
        self._bos = bos
        self._max_tokens = self._model.config.max_position_embeddings
        if _FREQUENCY_BUDGET + self._model.config.vocab_size >= MAX_TOTAL:
            raise ValueError("Model vocabulary exceeds the arithmetic coder's frequency limit")

    def _decode_text(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _predict(self, token: int, cache: Cache | None) -> tuple[NDArray[np.int32], Cache]:
        if cache is None and token == self._bos:
            cdf, start_cache = self._start_prediction
            # The model mutates the cache. Keep the BOS snapshot for other calls.
            return cdf, deepcopy(start_cache)
        return self._predict_uncached(token, cache)

    @cached_property
    def _start_prediction(self) -> tuple[NDArray[np.int32], Cache]:
        return self._predict_uncached(self._bos, None)

    def _predict_uncached(self, token: int, cache: Cache | None) -> tuple[NDArray[np.int32], Cache]:
        # Use the same single-token forward pass on both sides.
        # A batched pass can round differently from incremental decoding.
        ids = torch.tensor([[token]], device=self._model.device)
        output = self._model(input_ids=ids, past_key_values=cache, use_cache=True)
        logits = output.logits[0, -1].to(device="cpu", dtype=torch.float32)
        probabilities = logits.softmax(dim=-1)
        # Reserve one count per token, even where softmax rounds to zero.
        # Flooring the remaining counts keeps the total within the budget.
        counts = (probabilities * _FREQUENCY_BUDGET).to(torch.int32).add_(1)
        cdf = torch.empty(counts.numel() + 1, dtype=torch.int32)
        cdf[0] = 0
        torch.cumsum(counts, dim=0, out=cdf[1:])
        return cdf.numpy(), output.past_key_values

    @torch.inference_mode()
    def compress(self, text: str) -> tuple[bytes, int]:
        """Return the byte-packed arithmetic bits and the token count."""
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        if self._decode_text(tokens) != text:
            raise ValueError("This tokenizer cannot reproduce the input text exactly")
        self._check_length(len(tokens))
        if not tokens:
            return b"", 0

        encoder = ArithmeticEncoder()
        previous, cache = self._bos, None
        for token in tokens:
            cdf, cache = self._predict(previous, cache)
            encoder.write(cdf, token)
            previous = token
        return encoder.finish(), len(tokens)

    @torch.inference_mode()
    def decompress(self, bits: bytes | memoryview, token_count: int) -> str:
        """Recover text from byte-packed arithmetic bits and a token count."""
        self._check_length(token_count)
        if bool(token_count) != bool(bits):
            raise ValueError("Arithmetic payload does not match its token count")

        tokens = []
        if token_count:
            decoder = ArithmeticDecoder(bits)
            previous, cache = self._bos, None
            for _ in range(token_count):
                cdf, cache = self._predict(previous, cache)
                previous = decoder.read(cdf)
                tokens.append(previous)
        return self._decode_text(tokens)

    @cached_property
    def _cover_vocab(self) -> tuple[list[str], NDArray[np.intp]]:
        tokenizer = self._tokenizer
        surfaces = tokenizer.batch_decode(
            [[token] for token in range(self._model.config.vocab_size)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        special = set(tokenizer.all_special_ids)
        ids = np.array([
            token for token, text in enumerate(surfaces)
            if token not in special and text and "\ufffd" not in text
            and all(c.isprintable() or c in "\n\t" for c in text)
        ], dtype=np.intp)
        return surfaces, ids

    @staticmethod
    def _top_ranks(weights: NDArray[np.int32], size: int) -> NDArray[np.intp]:
        """Rank the heaviest tokens without sorting the entire vocabulary."""
        cutoff = np.partition(weights, -size)[-size]
        above_cutoff = np.flatnonzero(weights > cutoff)
        at_cutoff = np.flatnonzero(weights == cutoff)
        # Indices follow token ID order; lower IDs win ties at the cutoff too.
        top = np.concatenate((above_cutoff, at_cutoff[:size - len(above_cutoff)]))
        return top[np.lexsort((top, -weights[top]))]

    def _canonical_cover_tokens(
        self, candidates: NDArray[np.intp], prefix: list[int], text: str
    ) -> list[int]:
        """Keep only extensions that re-tokenize without changing the prefix."""
        surfaces, _ = self._cover_vocab
        candidate_ids = candidates.tolist()
        extended_texts = [text + surfaces[token] for token in candidate_ids]
        tokenizations = self._tokenizer(extended_texts, add_special_tokens=False)["input_ids"]
        return [
            token
            for token, tokens in zip(candidate_ids, tokenizations, strict=True)
            if len(tokens) == len(prefix) + 1 and tokens[-1] == token and tokens[:-1] == prefix
        ]

    def _cover_options(
        self, cdf: NDArray[np.int32], prefix: list[int], text: str
    ) -> tuple[list[int], list[int]]:
        """Build the same ranked, restricted distribution for rendering and recovery."""
        _, valid_ids = self._cover_vocab
        weights = np.diff(cdf)[valid_ids]
        batch_size = min(_COVER_CHOICES, len(valid_ids))
        if batch_size < 2:
            raise RenderError("Tokenizer has fewer than two printable cover tokens")

        top = self._top_ranks(weights, batch_size)
        choices = self._canonical_cover_tokens(valid_ids[top], prefix, text)
        if len(choices) < 2:
            # Only scan lower-ranked candidates when the top batch cannot encode bits.
            remaining = np.ones(len(valid_ids), dtype=bool)
            remaining[top] = False
            ranks = np.flatnonzero(remaining)
            ranks = ranks[np.lexsort((ranks, -weights[ranks]))]
            for start in range(0, len(ranks), batch_size):
                batch = ranks[start:start + batch_size]
                choices.extend(self._canonical_cover_tokens(valid_ids[batch], prefix, text))
                if len(choices) >= 2:
                    break
        if len(choices) < 2:
            raise RenderError("No reversible continuation with two token choices")

        choices = choices[:_COVER_CHOICES]
        counts = [int(cdf[token + 1]) - int(cdf[token]) for token in choices]
        # No token may hold more than half the mass; avoid near-zero bit progress.
        counts[0] = min(counts[0], sum(counts[1:]))
        return choices, [0, *np.cumsum(counts).tolist()]

    def _cover_start(self, context_tokens: list[int]) -> tuple[int, Cache | None]:
        if not context_tokens:
            return self._bos, None
        # Prefill BOS and all but the last context token. The first cover
        # prediction consumes that last token; it is not part of the cover.
        ids = torch.tensor([[self._bos, *context_tokens[:-1]]], device=self._model.device)
        output = self._model(input_ids=ids, use_cache=True, logits_to_keep=1)
        return context_tokens[-1], output.past_key_values

    @torch.inference_mode()
    def render_bits(self, bits: bytes, *, context: str = "") -> str:
        """Render ciphertext as a continuation of optional shared context."""
        if not bits:
            return ""
        context_tokens = self._tokenizer.encode(context, add_special_tokens=False) if context else []
        self._check_length(len(context_tokens) + 1)
        interval = PrefixInterval.from_bytes(bits)
        surfaces, _ = self._cover_vocab
        tokens = []
        text = ""
        previous_token, cache = self._cover_start(context_tokens)
        available_tokens = self._max_tokens - len(context_tokens)
        for _ in range(available_tokens):
            cdf, cache = self._predict(previous_token, cache)
            choices, cover_cdf = self._cover_options(cdf, tokens, text)
            cover_symbol = interval.read(cover_cdf)
            previous_token = choices[cover_symbol]
            tokens.append(previous_token)
            text += surfaces[previous_token]
            if interval.contained:
                return text
        raise RenderError(f"Ciphertext did not fit in the model context ({len(bits)} bytes)")

    @torch.inference_mode()
    def recover_bits(self, text: str, *, context: str = "") -> bytes:
        """Recover settled cover bytes using the same optional context."""
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        context_tokens = self._tokenizer.encode(context, add_special_tokens=False) if context else []
        self._check_length(len(context_tokens) + len(tokens))
        if not tokens:
            if text:
                raise ValueError("Not canonical cover text")
            return b""
        surfaces, _ = self._cover_vocab
        encoder = ArithmeticEncoder()
        prefix = []
        cover_offset = 0
        previous_token, cache = self._cover_start(context_tokens)
        for token in tokens:
            cdf, cache = self._predict(previous_token, cache)
            choices, cover_cdf = self._cover_options(cdf, prefix, text[:cover_offset])
            if token not in choices or not text.startswith(surfaces[token], cover_offset):
                raise ValueError("Not canonical cover text")
            cover_symbol = choices.index(token)
            encoder.write(cover_cdf, cover_symbol)
            prefix.append(token)
            cover_offset += len(surfaces[token])
            previous_token = token
        if cover_offset != len(text):
            raise ValueError("Not canonical cover text")
        return encoder.settled_bytes()

    def _check_length(self, count: int) -> None:
        if count < 0:
            raise ValueError("Token count cannot be negative")
        if count > self._max_tokens:
            raise ValueError(f"Text exceeds the model context limit of {self._max_tokens} tokens")
