import unittest
from tempfile import TemporaryDirectory

import torch
from tokenizers import Tokenizer, decoders, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from semantic_encrypting.codec import TextCodec
from semantic_encrypting.encryption import SemanticEncryptor, _open_message, _seal_message


class EnvelopeTests(unittest.TestCase):
    key = bytes(32)
    context = b"shared context"

    def test_end_marker_ignores_arithmetic_tail(self):
        message = bytes(1024)
        packed = _seal_message(message, self.key, self.context)
        self.assertEqual(
            _open_message(packed + b"\xff\x00\x02", self.key, self.context),
            message,
        )

    def test_missing_end_marker_is_rejected(self):
        packed = _seal_message(b"a", self.key, self.context)
        for truncated in (packed[:-2], packed[:-1]):
            with (
                self.subTest(truncated=truncated),
                self.assertRaisesRegex(ValueError, "missing its end marker"),
            ):
                _open_message(truncated, self.key, self.context)

    def test_invalid_escape_is_rejected(self):
        packed = _seal_message(b"a", self.key, self.context)
        with self.assertRaisesRegex(ValueError, "Invalid encrypted document escape"):
            _open_message(packed[:-1] + b"\x02", self.key, self.context)

    def test_changed_ciphertext_does_not_authenticate(self):
        packed = _seal_message(bytes(111), self.key, self.context)
        with self.assertRaisesRegex(ValueError, "modified encrypted document"):
            _open_message(b"\x01" + packed, self.key, self.context)

    def test_repeated_encryption_is_deterministic_and_bound_to_context(self):
        message = b"test"
        packed = _seal_message(message, self.key, self.context)
        self.assertEqual(_seal_message(message, self.key, self.context), packed)
        with self.assertRaisesRegex(ValueError, "modified encrypted document"):
            _open_message(packed, self.key, b"different context")


class PlaintextTerminationTests(unittest.TestCase):
    max_positions = 128

    @classmethod
    def setUpClass(cls):
        with TemporaryDirectory() as checkpoint:
            vocab = {token: i for i, token in enumerate(("<s>", "</s>", *"abcdefgh"))}
            tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
            tokenizer.decoder = decoders.Fuse()
            PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                bos_token="<s>",
                eos_token="</s>",
            ).save_pretrained(checkpoint)
            config = LlamaConfig(
                vocab_size=len(vocab),
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=1,
                bos_token_id=vocab["<s>"],
                eos_token_id=vocab["</s>"],
                max_position_embeddings=cls.max_positions,
            )
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(0)
                LlamaForCausalLM(config).save_pretrained(checkpoint)
            cls.codec = TextCodec(checkpoint)

    def test_cover_round_trip_with_empty_and_literal_eos_plaintext(self):
        encryptor = SemanticEncryptor(self.codec)
        key = bytes(32)
        for text in ("", "ab</s>cd"):
            with self.subTest(text=text):
                cover = encryptor.encrypt(text, key)
                self.assertEqual(encryptor.decrypt(cover, key), text)

    def test_plaintext_limit_reserves_one_position_for_termination(self):
        text = "a" * (self.max_positions - 1)
        self.assertEqual(self.codec.decompress(self.codec.compress(text)), text)
        with self.assertRaisesRegex(ValueError, "model context limit"):
            self.codec.compress(text + "a")

    def test_payload_without_end_symbol_stops_at_context_limit(self):
        # The zero code point always selects the first vocabulary symbol, not EOS.
        with self.assertRaisesRegex(ValueError, "Plaintext end symbol not found"):
            self.codec.decompress(b"\x00")
