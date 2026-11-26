# Semantic Encrypting

A notebook experiment in round-trip semantic encryption: turn plaintext into model-generated text, then recover the original plaintext exactly.

## Run the notebook

```sh
uv sync
uv run --no-sync jupyter notebook Main.ipynb
```

`uv sync` installs PyTorch with the project. If you need a device-specific build, install it in the project environment after syncing and keep `--no-sync` when launching the notebook or benchmark. Point the first cell in [`Main.ipynb`](Main.ipynb) at a local `LFM3-1.7B-Base` checkpoint. The notebook uses CUDA; change its device to `cpu` if needed. Run the cells in order to produce a plain-text document and recover the original message.

## Python API

From a checkout, run `uv sync` before using the API.

```python
from secrets import token_bytes

from semantic_encrypting.codec import TextCodec
from semantic_encrypting.encryption import SemanticEncryptor

encryptor = SemanticEncryptor(TextCodec(checkpoint_path, device="cuda:0"))
key = token_bytes(32)
document = encryptor.encrypt("Meet at noon.", key)
assert encryptor.decrypt(document, key) == "Meet at noon."
```

The output is one continuous cover text. There are no separators, footers, or external length fields. Pass the complete raw text back unchanged, including whitespace.

Pass the same optional `context` to both calls to make the cover a continuation of a shared model prefix:

```python
context = "The research vessel recorded these observations in its daily log:\n"
document = encryptor.encrypt("Meet at noon.", key, context=context)
assert encryptor.decrypt(document, key, context=context) == "Meet at noon."
```

The prefix is not included in the cover. Share it separately and preserve its exact text, including whitespace. It conditions cover generation and is authenticated, but it is not a secret key. Omit it to keep the original context-free behavior.

Encryption always uses nonce-free AES-SIV with a key derived from the supplied 32-byte key. The same message, key, context, and model environment produce identical cover text. **This reveals repeated messages.**

For example, this demo uses a fixed public key to make the cover repeatable:

```python
demo_key = bytes(range(32))  # Example only; use a random secret key in practice.
print(encryptor.encrypt("Meet at noon.", demo_key))
```

With `LFM3-1.7B-Base` on CUDA, the cover began:

```text
# 8 de mê guérilyè
- Guérilye, rèpublicen, anpi et pi gôches, 1
```

## The round trip

1. Arithmetic-code the plaintext and seal the bytes and token count with AES-SIV. The cover text encodes the authenticated sealed byte length as an unsigned LEB128 varint (one byte up to 127, two up to 16,383), followed by the sealed message. Equal lengths still have identical encodings.
2. Render the whole envelope in one pass. At each step, the existing tokenizer checks whether a candidate extends the exact token prefix. The arithmetic coder uses only those candidates and caps the largest probability at 50%.
3. Re-tokenize the cover and rebuild the same restricted probability tables to recover the envelope. Authenticate and decrypt it before decoding the plaintext.

Use the same model, tokenizer, and numerical environment on both sides. The renderer uses the checkpoint's configured context limit (262,144 tokens for the local LFM3-1.7B-Base checkpoint). The plaintext must fit on its own, and the optional prefix and cover must fit together. Generated cover text may include arbitrary markup and is not guaranteed to be fluent or undetectable.

The model's context window is a token limit, not a practical cover-size target. Both rendering and recovery check candidates against the full cover prefix because a tokenizer can revise earlier tokens when text is appended. Candidate validation can therefore scale quadratically with cover length. Keep messages short and use the benchmark below to establish a practical size limit for your checkpoint, device, and latency budget.

## Code map

- [`arithmetic.py`](src/semantic_encrypting/arithmetic.py) manages integer intervals and bit packing. Plaintext decoding follows one code point; cover generation follows an interval until it fixes the target byte prefix.
- [`codec.py`](src/semantic_encrypting/codec.py) turns model predictions into frequency tables. Plaintext uses the full vocabulary; cover rendering and recovery share the same tokenizer-checked candidate selection.
- [`encryption.py`](src/semantic_encrypting/encryption.py) seals and opens the binary envelope. The public methods connect it to plaintext compression and cover rendering.

## Benchmark rendering

```sh
uv run --no-sync python benchmark_render.py /path/to/LFM3-1.7B-Base \
  --input README.md
```

The benchmark checks exact recovery and reports render, encryption, and decryption times. Add `--context 'Shared model prefix'` to benchmark a prefix or `--device cpu` to run without CUDA.

## Test the encryption envelope

```sh
uv run --no-sync python -m unittest discover -s tests -p test_encryption.py
```

Run after `uv sync`. The tests do not load a model checkpoint.

## Type checking

With the project environment set up, run `uv tool install pyright` and then `pyright` from the repository root. Pyright uses the project's `.venv`.

## Credits

- Chen et al., [*Semantic Encryption: Secure and Effective Interaction with Cloud-based Large Language Models via Semantic Transformation*](https://arxiv.org/abs/2508.01638) (2025 preprint). Related work on preserving the logic of a prompt while changing its subject; this project instead encodes ciphertext for exact recovery.
- Ziegler, Deng, and Rush, [*Neural Linguistic Steganography*](https://aclanthology.org/D19-1115/) (2019), especially §4.4. Prior work on using a language model and arithmetic coding to turn message bits into reversible cover text, including generation conditioned on a shared context.

## License

The code in this repository is released under the [MIT License](LICENSE). Model weights and tokenizers retain their own licenses.
