import unittest

from semantic_encrypting.encryption import _open_message, _seal_message


class EnvelopeLengthTests(unittest.TestCase):
    key = bytes(32)
    context = b"shared context"

    def test_varint_boundaries_round_trip(self):
        for size, prefix in (
            (127, b"\x7f"),
            (128, b"\x80\x01"),
            (16_383, b"\xff\x7f"),
            (16_384, b"\x80\x80\x01"),
        ):
            with self.subTest(size=size):
                message = bytes(size - 16)
                packed = _seal_message(message, self.key, self.context)
                self.assertEqual(packed, prefix + packed[-size:])
                self.assertEqual(
                    _open_message(packed + b"\xff", self.key, self.context),
                    message,
                )

    def test_malformed_lengths_are_rejected(self):
        for header, error in (
            (b"\x80", "incomplete"),
            (b"\x94\x00" + bytes(20), "Invalid encrypted document length"),
            (b"\x94\x80\x80\x80\x10" + bytes(20), "Invalid encrypted document length"),
            (b"\x94\x80\x80\x80\x80\x00" + bytes(20), "Invalid encrypted document length"),
        ):
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, error):
                _open_message(header, self.key, self.context)

    def test_changed_length_does_not_authenticate(self):
        packed = _seal_message(bytes(111), self.key, self.context)
        tampered = b"\x7e" + packed[1:]
        with self.assertRaisesRegex(ValueError, "modified encrypted document"):
            _open_message(tampered, self.key, self.context)

    def test_repeated_encryption_is_deterministic_and_bound_to_context(self):
        message = b"\x00\x00\x00\x01test"
        packed = _seal_message(message, self.key, self.context)
        self.assertEqual(_seal_message(message, self.key, self.context), packed)
        with self.assertRaisesRegex(ValueError, "modified encrypted document"):
            _open_message(packed, self.key, b"different context")
