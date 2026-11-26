"""Turn plaintext into a plain-text document and recover it with a shared key."""

from struct import Struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESSIV
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .codec import TextCodec

_TAG_SIZE = 16
_UINT32 = Struct(">I")
_SIV_DOMAIN = b"semantic-encrypting/v4/deterministic-aes-siv"

_MARKER = b"\x00"
_ESCAPED_MARKER = b"\x00\x01"
_END = b"\x00\x00"


def _siv_cipher(key: bytes) -> AESSIV:
    derived = HKDF(algorithm=hashes.SHA256(), length=64, salt=None, info=_SIV_DOMAIN).derive(key)
    return AESSIV(derived)


def _unframe(packed: bytes) -> bytes:
    sealed = bytearray()
    start = 0
    while (marker := packed.find(_MARKER, start)) != -1 and marker + 1 < len(packed):
        suffix = packed[marker + 1]
        if suffix == _END[1]:
            # The final cover token can settle bits beyond the end marker.
            if not start:
                return packed[:marker]
            sealed.extend(packed[start:marker])
            return bytes(sealed)
        if suffix != _ESCAPED_MARKER[1]:
            raise ValueError("Invalid encrypted document escape")
        sealed.extend(packed[start:marker + 1])
        start = marker + 2
    raise ValueError("Encrypted document is missing its end marker")


def _seal_message(message: bytes, key: bytes, context: bytes) -> bytes:
    sealed = _siv_cipher(key).encrypt(message, [_SIV_DOMAIN, context])
    return sealed.replace(_MARKER, _ESCAPED_MARKER) + _END


def _open_message(packed: bytes, key: bytes, context: bytes) -> bytes:
    sealed = _unframe(packed)
    if len(sealed) < _TAG_SIZE + _UINT32.size:
        raise ValueError("Invalid encrypted document length")

    try:
        return _siv_cipher(key).decrypt(sealed, [_SIV_DOMAIN, context])
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
