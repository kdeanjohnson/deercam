"""Encrypts what DeerCam has to keep but nobody should be able to read off the disk: the Reveal
sign-in tokens.

The key comes from DEERCAM_SECRET_KEY, else DATA_DIR/.secret_key, made on first use. On a server, put
the key in the environment instead, so a backup of DATA_DIR doesn't carry the key along with the data.
Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
import json
import logging
import os
import threading

from cryptography.fernet import Fernet, InvalidToken

from .config import ROOT

log = logging.getLogger(__name__)
_lock = threading.Lock()
_fernet: Fernet | None = None


def _key() -> bytes:
    if os.environ.get("DEERCAM_SECRET_KEY"):
        return os.environ["DEERCAM_SECRET_KEY"].encode()
    path = ROOT / ".secret_key"
    if not path.exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Fernet.generate_key())
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        log.info("made a new encryption key in %s", path)
    return path.read_bytes().strip()


def _f() -> Fernet:
    global _fernet
    with _lock:
        if _fernet is None:
            _fernet = Fernet(_key())
        return _fernet


def seal(data: dict) -> str:
    return _f().encrypt(json.dumps(data).encode()).decode()


def unseal(token: str | None) -> dict | None:
    """None when there's nothing, or it was sealed with another key (the key changed)."""
    if not token:
        return None
    try:
        return json.loads(_f().decrypt(token.encode()))
    except InvalidToken:
        log.warning("can't decrypt saved Reveal tokens (did the secret key change?)")
        return None
