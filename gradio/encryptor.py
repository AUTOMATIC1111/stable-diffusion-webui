from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Hash import SHA256


def get_key(password: str) -> bytes:
    """Generates an encryption key based on the password provided."""
    key = SHA256.new(password.encode()).digest()
    return key


def encrypt(key: bytes, source: bytes) -> bytes:
    """Encrypts source data using the provided encryption key"""
    IV = Random.new().read(AES.block_size)  # generate IV
    encryptor = AES.new(key, AES.MODE_CBC, IV)
    padding = AES.block_size - len(source) % AES.block_size  # calculate needed padding
    source += bytes([padding]) * padding  # Python 2.x: source += chr(padding) * padding
    data = IV + encryptor.encrypt(source)  # store the IV at the beginning and encrypt
    return data


def decrypt(key: bytes, source: bytes) -> bytes:
    IV = source[: AES.block_size]  # extract the IV from the beginning
    decryptor = AES.new(key, AES.MODE_CBC, IV)
    data = decryptor.decrypt(source[AES.block_size :])  # decrypt
    padding = data[-1]  # pick the padding value from the end; Python 2.x: ord(data[-1])
    if (
        data[-padding:] != bytes([padding]) * padding
    ):  # Python 2.x: chr(padding) * padding
        raise ValueError("Invalid padding...")
    return data[:-padding]  # remove the padding
