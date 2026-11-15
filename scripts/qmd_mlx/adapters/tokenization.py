"""
tokenization.py — Single-pass tokenization, length-aware batches, exact order tracking
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from numbers import Integral
from typing import Any, List, Optional
import numpy as np

from ..protocol import InvalidInputError


@dataclass
class TokenizedBatch:
    """Unpadded token IDs, exact lengths, and indices in the original request."""

    token_ids: List[List[int]]
    lengths: List[int]
    original_indices: List[int]
    pad_token_id: int
    texts: Optional[List[str]] = None

    def __len__(self) -> int:
        return len(self.token_ids)

    def slice(self, indices: List[int]) -> TokenizedBatch:
        """Slice sequences without tokenizing or changing their original indices."""
        return TokenizedBatch(
            token_ids=[self.token_ids[i] for i in indices],
            lengths=[self.lengths[i] for i in indices],
            original_indices=[self.original_indices[i] for i in indices],
            pad_token_id=self.pad_token_id,
            texts=[self.texts[i] for i in indices] if self.texts is not None else None,
        )

    def pad_micro_batch(
        self,
        padding_side: str = "right",
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Pad only this microbatch, returning IDs, attention mask, and lengths."""
        if padding_side not in {"right", "left"}:
            raise InvalidInputError("padding_side must be 'right' or 'left'")
        if len(self.lengths) != len(self) or len(self.original_indices) != len(self):
            raise InvalidInputError("Malformed token batch: inconsistent row counts")
        for seq, length in zip(self.token_ids, self.lengths):
            if not isinstance(length, Integral) or isinstance(length, bool) or length != len(seq) or length <= 0:
                raise InvalidInputError("Malformed token batch: invalid token length")
        if not self.token_ids:
            return np.empty((0, 0), dtype=np.int32), np.empty((0, 0), dtype=np.int32), []

        max_len = max(self.lengths)
        padded_ids = np.full((len(self), max_len), self.pad_token_id, dtype=np.int32)
        attn_mask = np.zeros((len(self), max_len), dtype=np.int32)
        for i, (seq, length) in enumerate(zip(self.token_ids, self.lengths)):
            start = 0 if padding_side == "right" else max_len - length
            padded_ids[i, start : start + length] = seq
            attn_mask[i, start : start + length] = 1
        return padded_ids, attn_mask, list(self.lengths)


def _keyword_options(function: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Choose supported keywords *before* invoking a tokenizer, never retry it.

    A TypeError can occur after tokenization has already happened. Retrying with
    fewer arguments would encode that input twice and can hide tokenizer bugs.
    """
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return {}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return options
    return {
        name: value for name, value in options.items()
        if name in parameters and parameters[name].kind != inspect.Parameter.POSITIONAL_ONLY
    }


class TokenizerHelper:
    """Use a genuine unpadded batch API, or an exactly-once encode-only API."""

    @staticmethod
    def get_pad_token_id(tokenizer: Any) -> int:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        if isinstance(pad_id, bool) or not isinstance(pad_id, Integral) or not 0 <= pad_id <= np.iinfo(np.int32).max:
            raise InvalidInputError("Tokenizer returned an invalid pad token ID")
        return int(pad_id)

    @staticmethod
    def _validate_ids(ids: Any, index: int, max_length: int) -> list[int]:
        if not isinstance(ids, (list, tuple, np.ndarray)) or (isinstance(ids, np.ndarray) and ids.ndim != 1):
            raise InvalidInputError(f"Tokenizer returned invalid token IDs at index {index}")
        if len(ids) == 0:
            raise InvalidInputError(f"Tokenizer returned no tokens at index {index}")
        if len(ids) > max_length:
            raise InvalidInputError(
                f"Text at index {index} token length ({len(ids)}) exceeds max_length ({max_length})"
            )
        if any(isinstance(token, (bool, np.bool_)) or not isinstance(token, Integral)
               or not 0 <= token <= np.iinfo(np.int32).max for token in ids):
            raise InvalidInputError(f"Tokenizer returned invalid token IDs at index {index}")
        # Own the lists: mutable tokenizer buffers must not change queued work.
        return [int(token) for token in ids]

    @staticmethod
    def tokenize_once(
        texts: List[str],
        raw_tokenizer: Any,
        max_length: int = 2048,
    ) -> TokenizedBatch:
        """Encode every text once, without padding, truncation, or order changes."""
        if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length <= 0:
            raise InvalidInputError("max_length must be a positive integer")
        if not isinstance(texts, (list, tuple)):
            raise InvalidInputError("texts must be a list of strings")
        # Validate the whole request before doing any tokenization work.
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise InvalidInputError(f"Text at index {i} is empty, whitespace, or not a string")
        pad_id = TokenizerHelper.get_pad_token_id(raw_tokenizer)
        owned_texts = list(texts)
        if not texts:
            return TokenizedBatch([], [], [], pad_id, texts=[])

        options = {"add_special_tokens": True, "padding": False, "truncation": False}
        batch_function = None
        batch_options = {}
        # Prefer HF's batched __call__; batch_encode_plus also supports older
        # implementations with no callable interface. Only select an API whose
        # signature lets us explicitly disable padding AND truncation.
        for candidate in (raw_tokenizer, getattr(raw_tokenizer, "batch_encode_plus", None)):
            if callable(candidate):
                candidate_options = _keyword_options(candidate, options)
                if "padding" in candidate_options and "truncation" in candidate_options:
                    batch_function, batch_options = candidate, candidate_options
                    break
        try:
            if batch_function is not None:
                result = batch_function(owned_texts, **batch_options)
                if not isinstance(result, Mapping) or "input_ids" not in result:
                    raise InvalidInputError("Batch tokenizer returned no input_ids")
                rows = result["input_ids"]
                if not isinstance(rows, (list, tuple, np.ndarray)) or len(rows) != len(texts):
                    raise InvalidInputError("Batch tokenizer returned an invalid token row count")
                masks = result.get("attention_mask")
                if masks is not None:
                    if len(masks) != len(rows) or any(
                        len(mask) != len(row) or any(value != 1 for value in mask)
                        for mask, row in zip(masks, rows)
                    ):
                        raise InvalidInputError("Batch tokenizer returned padded or invalid token masks")
            else:
                encode = getattr(raw_tokenizer, "encode", None)
                if not callable(encode):
                    raise InvalidInputError("Tokenizer has no supported batch or encode API")
                encode_options = _keyword_options(encode, {"add_special_tokens": True, "truncation": False})
                rows = [encode(text, **encode_options) for text in owned_texts]
            token_ids = [TokenizerHelper._validate_ids(row, i, max_length) for i, row in enumerate(rows)]
        except InvalidInputError:
            raise
        except Exception as exc:
            # Do not fall back after an attempted batch call or retry an encode:
            # it could have already performed some or all of the work.
            raise InvalidInputError(f"Tokenization failed: {exc}") from exc

        return TokenizedBatch(
            token_ids=token_ids,
            lengths=[len(ids) for ids in token_ids],
            original_indices=list(range(len(texts))),
            pad_token_id=pad_id,
            texts=owned_texts,
        )
