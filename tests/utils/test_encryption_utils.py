import os
import tempfile
import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet, InvalidToken

from data_juicer.utils.encryption_utils import (
    decrypt_file_to_bytes,
    decrypt_file_to_bytesio,
    encrypt_file,
    get_secure_tmpdir,
    load_fernet_key,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

_ENV_KEY = "DJ_ENCRYPTION_KEY"


class LoadFernetKeyTest(DataJuicerTestCaseBase):
    """Tests for load_fernet_key()."""

    def setUp(self):
        super().setUp()
        self.key = Fernet.generate_key()
        # Ensure the env var is clean before each test
        os.environ.pop(_ENV_KEY, None)

    def tearDown(self):
        os.environ.pop(_ENV_KEY, None)
        super().tearDown()

    def test_load_from_key_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(self.key)
            key_path = f.name
        try:
            fernet = load_fernet_key(key_path)
            # Round-trip sanity check
            token = fernet.encrypt(b"hello")
            self.assertEqual(fernet.decrypt(token), b"hello")
        finally:
            os.remove(key_path)

    def test_load_from_env_var(self):
        os.environ[_ENV_KEY] = self.key.decode()
        fernet = load_fernet_key(None)
        token = fernet.encrypt(b"world")
        self.assertEqual(fernet.decrypt(token), b"world")

    def test_key_file_takes_priority_over_env(self):
        """key_path should shadow any env var."""
        other_key = Fernet.generate_key()
        os.environ[_ENV_KEY] = other_key.decode()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(self.key)
            key_path = f.name
        try:
            fernet = load_fernet_key(key_path)
            # The fernet loaded from the file can decrypt what self.key encrypted
            plain = b"priority test"
            token = Fernet(self.key).encrypt(plain)
            self.assertEqual(fernet.decrypt(token), plain)
        finally:
            os.remove(key_path)

    def test_missing_key_raises(self):
        """No key_path and no env var should raise ValueError."""
        with self.assertRaises(ValueError):
            load_fernet_key(None)

    def test_nonexistent_key_file_raises(self):
        with self.assertRaises(ValueError, msg="encryption_key_path"):
            load_fernet_key("/nonexistent/path/to/key.key")

    def test_invalid_key_content_raises(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".key") as f:
            f.write(b"not-a-valid-fernet-key")
            key_path = f.name
        try:
            with self.assertRaises(ValueError, msg="Invalid Fernet key"):
                load_fernet_key(key_path)
        finally:
            os.remove(key_path)


class EncryptFileTest(DataJuicerTestCaseBase):
    """Tests for encrypt_file()."""

    def setUp(self):
        super().setUp()
        self.fernet = Fernet(Fernet.generate_key())

    def test_encrypt_to_different_dst(self):
        plaintext = b"secret data"
        with tempfile.NamedTemporaryFile(delete=False) as src:
            src.write(plaintext)
            src_path = src.name
        with tempfile.NamedTemporaryFile(delete=False) as dst:
            dst_path = dst.name
        try:
            encrypt_file(src_path, dst_path, self.fernet)
            with open(dst_path, "rb") as f:
                ciphertext = f.read()
            # dst must be ciphertext
            self.assertNotEqual(ciphertext, plaintext)
            # must be decryptable
            self.assertEqual(self.fernet.decrypt(ciphertext), plaintext)
            # src must still contain original plaintext
            with open(src_path, "rb") as f:
                self.assertEqual(f.read(), plaintext)
        finally:
            os.remove(src_path)
            os.remove(dst_path)

    def test_encrypt_in_place(self):
        plaintext = b"in-place secret"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(plaintext)
            path = f.name
        try:
            encrypt_file(path, path, self.fernet)
            with open(path, "rb") as f:
                ciphertext = f.read()
            self.assertNotEqual(ciphertext, plaintext)
            self.assertEqual(self.fernet.decrypt(ciphertext), plaintext)
        finally:
            os.remove(path)


class DecryptFileToBytesTest(DataJuicerTestCaseBase):
    """Tests for decrypt_file_to_bytes()."""

    def setUp(self):
        super().setUp()
        self.fernet = Fernet(Fernet.generate_key())

    def test_roundtrip(self):
        plaintext = b"round-trip data"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(self.fernet.encrypt(plaintext))
            path = f.name
        try:
            result = decrypt_file_to_bytes(path, self.fernet)
            self.assertEqual(result, plaintext)
        finally:
            os.remove(path)

    def test_wrong_key_raises(self):
        plaintext = b"secret"
        wrong_fernet = Fernet(Fernet.generate_key())
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(self.fernet.encrypt(plaintext))
            path = f.name
        try:
            with self.assertRaises(InvalidToken):
                decrypt_file_to_bytes(path, wrong_fernet)
        finally:
            os.remove(path)


class DecryptFileToBytesIOTest(DataJuicerTestCaseBase):
    """Tests for decrypt_file_to_bytesio()."""

    def setUp(self):
        super().setUp()
        self.fernet = Fernet(Fernet.generate_key())

    def test_returns_seekable_buffer(self):
        plaintext = b"bytesio content"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(self.fernet.encrypt(plaintext))
            path = f.name
        try:
            buf = decrypt_file_to_bytesio(path, self.fernet)
            self.assertEqual(buf.read(), plaintext)
            # Must be seekable and positioned at 0
            buf.seek(0)
            self.assertEqual(buf.read(), plaintext)
        finally:
            os.remove(path)

    def test_name_attribute_is_basename(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            f.write(self.fernet.encrypt(b"{}"))
            path = f.name
        try:
            buf = decrypt_file_to_bytesio(path, self.fernet)
            self.assertEqual(buf.name, os.path.basename(path))
        finally:
            os.remove(path)


class GetSecureTmpdirTest(DataJuicerTestCaseBase):
    """Tests for get_secure_tmpdir()."""

    def test_returns_shm_when_available(self):
        with patch("os.path.isdir", return_value=True), \
             patch("os.access", return_value=True):
            result = get_secure_tmpdir()
        self.assertEqual(result, "/dev/shm")

    def test_returns_none_when_shm_not_a_dir(self):
        with patch("os.path.isdir", return_value=False):
            result = get_secure_tmpdir()
        self.assertIsNone(result)

    def test_returns_none_when_shm_not_writable(self):
        with patch("os.path.isdir", return_value=True), \
             patch("os.access", return_value=False):
            result = get_secure_tmpdir()
        self.assertIsNone(result)

    def test_actual_behaviour_does_not_raise(self):
        """get_secure_tmpdir() must never raise regardless of the platform."""
        result = get_secure_tmpdir()
        self.assertIn(result, ["/dev/shm", None])


if __name__ == "__main__":
    unittest.main()
