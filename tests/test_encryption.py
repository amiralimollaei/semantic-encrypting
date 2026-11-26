import unittest

from semantic_encrypting.encryption import _open_message, _seal_message


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
        packed = _seal_message(bytes(4), self.key, self.context)
        for truncated in (packed[:-2], packed[:-1]):
            with (
                self.subTest(truncated=truncated),
                self.assertRaisesRegex(ValueError, "missing its end marker"),
            ):
                _open_message(truncated, self.key, self.context)

    def test_invalid_escape_is_rejected(self):
        packed = _seal_message(bytes(4), self.key, self.context)
        with self.assertRaisesRegex(ValueError, "Invalid encrypted document escape"):
            _open_message(packed[:-1] + b"\x02", self.key, self.context)

    def test_changed_ciphertext_does_not_authenticate(self):
        packed = _seal_message(bytes(111), self.key, self.context)
        with self.assertRaisesRegex(ValueError, "modified encrypted document"):
            _open_message(b"\x01" + packed, self.key, self.context)

    def test_repeated_encryption_is_deterministic_and_bound_to_context(self):
        message = b"\x00\x00\x00\x01test"
        packed = _seal_message(message, self.key, self.context)
        self.assertEqual(_seal_message(message, self.key, self.context), packed)
        with self.assertRaisesRegex(ValueError, "modified encrypted document"):
            _open_message(packed, self.key, b"different context")
