import gzip
import hashlib
import os

import numpy as np
from PIL import Image

try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad, unpad
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Gzip header to detect if data is compressed
__GZIP_MAGIC_NUMBER_BYTES = b'\x1F\x8B\x08\x00'
# Data suffix so we can detect the end of the envelope
__IMPRINT_MAGIC_NUMBER_BYTES = b'\x0C\xAB\x00\x5E'

def preprocess_imprint_data(input_str: str, compress: bool = True, pass_phrase: str = '') -> bytes:
    """
    Preprocess the input data before imprinting onto an image.

    This function compresses and/or encrypts the input data based on the provided arguments.

    Args:
        input_str (str): The input string to be processed.
        compress (bool, optional): Indicates whether to compress the data using BZ3. Defaults to True.
        pass_phrase (str, optional): Passphrase for encrypting the data using AES. If empty, no encryption is applied. Defaults to ''.

    Returns:
        bytes: The preprocessed data as bytes, ready for imprinting onto an image.
    """
    complete_data = bytes(input_str.encode('utf-8'))
    
    if pass_phrase and not CRYPTO_AVAILABLE:
        raise ImportError("pycryptodome package is required for data encryption. Please install it before using this feature.")
    
    if compress:
        complete_data = gzip.compress(complete_data, compresslevel=9)
    
    if pass_phrase and CRYPTO_AVAILABLE:
        # If pass_phrase is not empty, then encrypt the data
        key = hashlib.sha256(pass_phrase.encode('utf-8')).digest()
        iv = os.urandom(AES.block_size)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        # Ensure that complete_data is padded to AES.block_size
        complete_data = pad(complete_data, AES.block_size)
        complete_data = iv + cipher.encrypt(complete_data)
    
    complete_data += __IMPRINT_MAGIC_NUMBER_BYTES
    
    return complete_data

def encode_imprint_data(input_image: Image, encode_data: bytes) -> Image.Image:
    """
    Encode the input data into the input image.

    This function imprints the input data onto the last 8 rows of pixels in the input image.

    Args:
        input_image (Image): The input image for encoding the data.
        encode_data (bytes): The preprocessed data to be imprinted onto the image.

    Returns:
        Image.Image: The resulting image with the encoded data.
    """
    input_image = input_image.convert('RGB')
    
    # Convert encode_data_final into a 1d numpy array, of type byte
    encode_data_final = np.frombuffer(encode_data, dtype=np.uint8)
    encode_data_final = np.unpackbits(encode_data_final)
    
    # Convert the image data to a numpy array and reshape it to be flat, R G R G B, so no (200, 3) instead (600, )
    image_np = np.array(input_image).reshape(-1)
    
    # Calculate the starting index of the last 8 rows of pixels.
    seek_ix = len(image_np) - 8 * (input_image.width * 3)
    
    # Get a view of the last 8 rows of pixels from the image
    last_8_start = image_np[seek_ix:seek_ix + len(encode_data_final)]
    
    # Mask off the last 8 rows of pixels so the data slate is clean
    last_8_start &= 0xFE
    last_8_start |= encode_data_final
    
    # Convert the numpy array back to a PIL image without reallocating a new image.
    input_image.frombytes(image_np)
    
    return input_image

def postprocess_imprint_data(input_bytes: bytes, pass_phrase: str = ''):
    """
    Postprocess the extracted data from an image.

    This function decrypts and/or decompresses the input data based on the provided arguments.

    Args:
        input_bytes (bytes): The extracted data bytes from the image.
        pass_phrase (str, optional): Passphrase for decrypting the data using AES. If empty, no decryption is applied. Defaults to ''.

    Returns:
        bytes: The postprocessed data as bytes, ready to be decoded into a string.
    """
    end_ix = input_bytes.find(__IMPRINT_MAGIC_NUMBER_BYTES)
    
    if end_ix == -1:
        # Not recognized as imprinted data
        return bytes(0)
    
    input_bytes = input_bytes[:end_ix]
    
    if pass_phrase and not CRYPTO_AVAILABLE:
        raise ImportError("pycryptodome package is required for data encryption. Please install it before using this feature.")
    
    if pass_phrase and CRYPTO_AVAILABLE:
        # If pass_phrase is not empty, then (try to) decrypt the data
        key = hashlib.sha256(pass_phrase.encode('utf-8')).digest()
        iv = input_bytes[:AES.block_size]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        input_bytes = cipher.decrypt(input_bytes[AES.block_size:end_ix])
        # Trim trailing 0x00 bytes (likely from padding)
        input_bytes = unpad(input_bytes, AES.block_size)
        # input_bytes = input_bytes.rstrip(b'\x00')
    
    if len(input_bytes) > 10 and input_bytes[0:4] == __GZIP_MAGIC_NUMBER_BYTES:
        return gzip.decompress(input_bytes)
    
    return input_bytes[:end_ix]

def decode_imprint_data(input_image: Image, pass_phrase: str = '') -> str:
    """
    Decode the imprinted data from the input image.

    This function extracts the data from the last 8 rows of pixels in the input image and postprocesses it.

    Args:
        input_image (Image): The input image with the imprinted data.
        pass_phrase (str, optional): Passphrase for decrypting the data using AES. If empty, no decryption is applied. Defaults to ''.

    Returns:
        str: The decoded string extracted from the input image.
    """
    image_np = np.array(input_image).reshape(-1)
    seek_ix = len(image_np) - 8 * (input_image.width * 3)
    image_np_last_8_rows = image_np[seek_ix:]
    # Mask off the last 8 rows of data so that each byte is only 0b0000000X
    image_np_last_8_rows &= 1
    # Convert the last bits spread across all the bytes back down to packed bytes. This basically just compacts 8 to 1
    recovered_bytes = np.packbits(image_np_last_8_rows).tobytes()
    recovered_bytes = postprocess_imprint_data(recovered_bytes, pass_phrase=pass_phrase)
    decoded_str = recovered_bytes.decode('utf-8') if recovered_bytes else ''
    return decoded_str

if __name__ == '__main__':
    pass
