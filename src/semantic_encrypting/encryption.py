"""Turn plaintext into a plain-text document and recover it with a shared key."""

from secrets import token_bytes
from struct import Struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESSIV, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .codec import TextCodec

_NONCE_SIZE = 12
_TAG_SIZE = 16
_UINT32 = Struct(">I")
_DOMAIN = b"semantic-encrypting/v2"
_SIV_DOMAIN = _DOMAIN+b"/deterministic-aes-siv"

# The cover carries header || sealed message. The header stores the optional
# nonce followed by the sealed length; the sealed message starts with the
# plaintext token count. Both integers are big-endian.


def _siv_cipher(key: bytes) -> AESSIV:
    derived = HKDF(algorithm=hashes.SHA256(), length=64, salt=None, info=_SIV_DOMAIN).derive(key)
    return AESSIV(derived)


def _seal_message(
    message: bytes, key: bytes, context: bytes, deterministic: bool
) -> bytes:
    nonce = b"" if deterministic else token_bytes(_NONCE_SIZE)
    header = nonce + _UINT32.pack(len(message) + _TAG_SIZE)
    if deterministic:
        sealed = _siv_cipher(key).encrypt(message, [_SIV_DOMAIN, header, context])
    else:
        associated_data = _DOMAIN + header + context
        sealed = ChaCha20Poly1305(key).encrypt(nonce, message, associated_data)
    return header + sealed


def _open_message(
    packed: bytes, key: bytes, context: bytes, deterministic: bool
) -> bytes:
    nonce_size = 0 if deterministic else _NONCE_SIZE
    header_size = nonce_size + _UINT32.size
    if len(packed) < header_size:
        raise ValueError("Encrypted document is incomplete")

    sealed_size, = _UINT32.unpack_from(packed, nonce_size)
    sealed_end = header_size + sealed_size
    if sealed_size < _TAG_SIZE + _UINT32.size or sealed_end > len(packed):
        raise ValueError("Invalid encrypted document length")

    header = packed[:header_size]
    # The final cover token can settle bits beyond the authenticated envelope.
    sealed = packed[header_size:sealed_end]
    try:
        if deterministic:
            return _siv_cipher(key).decrypt(sealed, [_SIV_DOMAIN, header, context])
        nonce = header[:nonce_size]
        associated_data = _DOMAIN + header + context
        return ChaCha20Poly1305(key).decrypt(nonce, sealed, associated_data)
    except InvalidTag as exc:
        raise ValueError("Wrong key or modified encrypted document") from exc


class SemanticEncryptor:
    def __init__(self, codec: TextCodec) -> None:
        self._codec = codec

    def encrypt(
        self, text: str, key: bytes, *, context: str = "", deterministic: bool = False
    ) -> str:
        """Encrypt text as one cover, randomized unless deterministic is requested."""
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes")
        compressed, token_count = self._codec.compress(text)
        message = _UINT32.pack(token_count) + compressed
        envelope = _seal_message(message, key, context.encode("utf-8"), deterministic)
        return self._codec.render_bits(envelope, context=context)

    def decrypt(
        self, document: str, key: bytes, *, context: str = "", deterministic: bool = False
    ) -> str:
        """Recover text with the same context and mode used to encrypt it."""
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes")
        packed = self._codec.recover_bits(document, context=context)
        message = _open_message(packed, key, context.encode("utf-8"), deterministic)
        token_count, = _UINT32.unpack_from(message)
        compressed = memoryview(message)[_UINT32.size:]
        return self._codec.decompress(compressed, token_count)
