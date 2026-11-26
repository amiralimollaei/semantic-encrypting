"""Turn plaintext into a plain-text document and recover it with a shared key."""

from struct import Struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESSIV
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .codec import TextCodec

_TAG_SIZE = 16
_UINT32 = Struct(">I")
_SIV_DOMAIN = b"semantic-encrypting/v3/deterministic-aes-siv"

# The cover carries the unsigned LEB128 sealed length followed by the sealed
# message, which starts with the big-endian plaintext token count.


def _siv_cipher(key: bytes) -> AESSIV:
    derived = HKDF(algorithm=hashes.SHA256(), length=64, salt=None, info=_SIV_DOMAIN).derive(key)
    return AESSIV(derived)


def _encode_sealed_size(size: int) -> bytes:
    if size >= 1 << 32:
        raise ValueError("Encrypted document is too large")
    encoded = bytearray()
    while size >= 0x80:
        encoded.append((size & 0x7F) | 0x80)
        size >>= 7
    encoded.append(size)
    return bytes(encoded)


def _decode_sealed_size(packed: bytes) -> tuple[int, int]:
    size = 0
    for offset, shift in enumerate(range(0, 32, 7)):
        if offset >= len(packed):
            raise ValueError("Encrypted document is incomplete")
        byte = packed[offset]
        if (shift == 28 and byte > 0x0F) or (shift and byte == 0):
            break
        size |= (byte & 0x7F) << shift
        if byte < 0x80:
            return size, offset + 1
    raise ValueError("Invalid encrypted document length")


def _seal_message(message: bytes, key: bytes, context: bytes) -> bytes:
    header = _encode_sealed_size(len(message) + _TAG_SIZE)
    sealed = _siv_cipher(key).encrypt(message, [_SIV_DOMAIN, header, context])
    return header + sealed


def _open_message(packed: bytes, key: bytes, context: bytes) -> bytes:
    sealed_size, header_size = _decode_sealed_size(packed)
    sealed_end = header_size + sealed_size
    if sealed_size < _TAG_SIZE + _UINT32.size or sealed_end > len(packed):
        raise ValueError("Invalid encrypted document length")

    header = packed[:header_size]
    # The final cover token can settle bits beyond the authenticated envelope.
    sealed = packed[header_size:sealed_end]
    try:
        return _siv_cipher(key).decrypt(sealed, [_SIV_DOMAIN, header, context])
    except InvalidTag as exc:
        raise ValueError("Wrong key or modified encrypted document") from exc


class SemanticEncryptor:
    def __init__(self, codec: TextCodec) -> None:
        self._codec = codec

    def encrypt(self, text: str, key: bytes, *, context: str = "") -> str:
        """Encrypt text as one deterministic cover."""
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes")
        compressed, token_count = self._codec.compress(text)
        message = _UINT32.pack(token_count) + compressed
        envelope = _seal_message(message, key, context.encode("utf-8"))
        return self._codec.render_bits(envelope, context=context)

    def decrypt(self, document: str, key: bytes, *, context: str = "") -> str:
        """Recover text with the same key and context used to encrypt it."""
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes")
        packed = self._codec.recover_bits(document, context=context)
        message = _open_message(packed, key, context.encode("utf-8"))
        token_count, = _UINT32.unpack_from(message)
        compressed = memoryview(message)[_UINT32.size:]
        return self._codec.decompress(compressed, token_count)
