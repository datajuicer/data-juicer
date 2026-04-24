import io
import os

from loguru import logger

# Environment variable name for the Fernet key (base64 url-safe string)
_DJ_ENCRYPTION_KEY_ENV = "DJ_ENCRYPTION_KEY"


def load_fernet_key(key_path=None):
    """
    Load a Fernet key from a file or environment variable.

    Priority order:
    1. ``key_path`` file (if provided and exists)
    2. Environment variable ``DJ_ENCRYPTION_KEY``

    :param key_path: path to a file containing the Fernet key as a
        base64 url-safe string.  Pass ``None`` to fall back to the
        environment variable.
    :return: a :class:`cryptography.fernet.Fernet` instance ready for
        encryption / decryption.
    :raises ValueError: if no key can be found or the key is invalid.
    """
    from cryptography.fernet import Fernet

    raw_key = None

    if key_path is not None:
        if not os.path.exists(key_path):
            raise ValueError(f"encryption_key_path '{key_path}' does not exist.")
        with open(key_path, "rb") as f:
            raw_key = f.read().strip()
        logger.debug(f"Loaded Fernet key from file: {key_path}")
    else:
        env_key = os.environ.get(_DJ_ENCRYPTION_KEY_ENV)
        if env_key:
            raw_key = env_key.strip().encode()
            logger.debug(f"Loaded Fernet key from environment variable " f"{_DJ_ENCRYPTION_KEY_ENV}.")

    if raw_key is None:
        raise ValueError(
            "No encryption key found. Provide 'encryption_key_path' in "
            f"config or set the environment variable {_DJ_ENCRYPTION_KEY_ENV}."
        )

    try:
        return Fernet(raw_key)
    except Exception as e:
        raise ValueError(f"Invalid Fernet key: {e}") from e


def encrypt_file(src_path, dst_path, fernet):
    """
    Encrypt a file with Fernet and write the ciphertext to ``dst_path``.

    When ``src_path == dst_path`` the file is encrypted in-place: the
    plaintext is read into memory, the file is overwritten with ciphertext,
    and the original plaintext is never written back to disk.

    :param src_path: path to the plaintext source file.
    :param dst_path: path where the encrypted file will be written.
        May be the same as ``src_path`` for in-place encryption.
    :param fernet: a :class:`cryptography.fernet.Fernet` instance.
    """
    with open(src_path, "rb") as f:
        plaintext = f.read()
    ciphertext = fernet.encrypt(plaintext)
    # Write atomically: for in-place case we overwrite after reading
    with open(dst_path, "wb") as f:
        f.write(ciphertext)
    logger.debug(f"Encrypted file written to: {dst_path}")


def decrypt_file_to_bytes(src_path, fernet):
    """
    Decrypt an encrypted file and return the plaintext as :class:`bytes`.

    The plaintext is **never written to disk** — only returned in memory.

    :param src_path: path to the Fernet-encrypted file.
    :param fernet: a :class:`cryptography.fernet.Fernet` instance.
    :return: decrypted plaintext as :class:`bytes`.
    :raises cryptography.fernet.InvalidToken: if the file cannot be
        decrypted with the provided key.
    """
    with open(src_path, "rb") as f:
        ciphertext = f.read()
    plaintext = fernet.decrypt(ciphertext)
    logger.debug(f"Decrypted file to memory: {src_path}")
    return plaintext


def get_secure_tmpdir():
    """Return the best available temporary directory for plaintext data.

    Priority:
    1. ``/dev/shm`` — Linux in-memory tmpfs, plaintext never touches disk.
    2. System default (``/tmp`` or ``TMPDIR``) — plaintext exists briefly on
       disk until the caller removes the file.

    :return: path string to use as the ``dir`` argument of
        :func:`tempfile.NamedTemporaryFile`.
    """
    import tempfile

    shm = "/dev/shm"
    if os.path.isdir(shm) and os.access(shm, os.W_OK):
        logger.debug(
            "Using /dev/shm as temporary directory for decrypted data "
            "(in-memory filesystem — plaintext never touches disk)."
        )
        return shm
    logger.debug(
        "'/dev/shm' is not available; falling back to the system temporary "
        "directory (%s). Decrypted plaintext will exist briefly on disk "
        "before being removed.",
        tempfile.gettempdir(),
    )
    return None  # None → tempfile uses the system default


def decrypt_file_to_bytesio(src_path, fernet):
    """
    Decrypt an encrypted file and return an :class:`io.BytesIO` buffer.

    Convenience wrapper around :func:`decrypt_file_to_bytes` that wraps
    the result in a seekable in-memory buffer, ready to be passed directly
    to HuggingFace ``load_dataset`` or PDF/DOCX parsers.

    :param src_path: path to the Fernet-encrypted file.
    :param fernet: a :class:`cryptography.fernet.Fernet` instance.
    :return: :class:`io.BytesIO` positioned at offset 0.
    """
    data = decrypt_file_to_bytes(src_path, fernet)
    buf = io.BytesIO(data)
    buf.name = os.path.basename(src_path)  # some parsers inspect .name
    return buf
