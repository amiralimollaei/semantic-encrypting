"""Measure unsplit cover rendering and exact recovery with the local checkpoint."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from transformers.cache_utils import Cache

from semantic_encrypting.codec import TextCodec
from semantic_encrypting.encryption import SemanticEncryptor


class MeasuredCodec(TextCodec):
    model_steps = 0
    render_seconds = 0.0

    def _predict_uncached(self, token: int, cache: Cache | None) -> tuple[NDArray[np.int32], Cache]:
        self.model_steps += 1
        return super()._predict_uncached(token, cache)

    def render_bits(self, bits: bytes, *, context: str = "") -> str:
        self.render_seconds = 0.0
        start = perf_counter()
        try:
            return super().render_bits(bits, context=context)
        finally:
            self.render_seconds += perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("README.md"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context", default="", help="Shared model prefix for the cover")
    args = parser.parse_args()

    samples = [
        ("short", "Meet at noon."),
        ("document", args.input.read_text(encoding="utf-8")),
    ]
    codec = MeasuredCodec(args.checkpoint, args.device)
    key = bytes(32)  # Benchmark only; never use a public key for private messages.
    encryptor = SemanticEncryptor(codec)
    codec.compress("Warm up.")

    print(
        "sample,input_bytes,render_seconds,encrypt_seconds,"
        "decrypt_seconds,encrypt_incremental_steps",
        flush=True,
    )
    for name, text in samples:
        codec.model_steps = 0
        encrypt_started = perf_counter()
        cover = encryptor.encrypt(text, key, context=args.context)
        encrypt_seconds = perf_counter() - encrypt_started
        encrypt_steps = codec.model_steps
        decrypt_started = perf_counter()
        restored = encryptor.decrypt(cover, key, context=args.context)
        decrypt_seconds = perf_counter() - decrypt_started
        if restored != text:
            raise AssertionError(f"Decrypted text does not match the {name} fixture")
        input_bytes = len(text.encode("utf-8"))
        print(
            f"{name},{input_bytes},{codec.render_seconds:.3f},"
            f"{encrypt_seconds:.3f},{decrypt_seconds:.3f},{encrypt_steps}",
            flush=True,
        )


if __name__ == "__main__":
    main()
