"""Token 加密解密器 - 使用 AES-CBC 生成/解析 sk- token"""

import base64
import hashlib
import json
import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class TokenEncryptor:
    """AES-CBC Token 加密解密器"""

    def __init__(self, secret_key: str):
        """
        初始化加密器

        Args:
            secret_key: 用于加密的密钥字符串
        """
        self.key = hashlib.sha256(secret_key.encode()).digest()

    def _pad(self, data: bytes) -> bytes:
        """PKCS7 填充"""
        padding_length = 16 - (len(data) % 16)
        return data + bytes([padding_length] * padding_length)

    def _unpad(self, data: bytes) -> bytes:
        """移除 PKCS7 填充"""
        return data[: -data[-1]]

    def encrypt_user_id(self, user_data: dict) -> str:
        """
        加密用户数据，生成 sk- token

        Args:
            user_data: 包含用户信息的字典 (必须包含 user_id 字段)

        Returns:
            以 'sk-' 开头的加密 token
        """
        json_str = json.dumps(user_data, ensure_ascii=False, separators=(",", ":"))
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(self._pad(json_str.encode())) + encryptor.finalize()
        token_body = base64.b64encode(iv + encrypted).decode().rstrip("=")
        return f"sk-{token_body}"

    def decrypt_token(self, token: str) -> dict | None:
        """
        解密 sk- token，获取原始用户数据

        Args:
            token: 以 'sk-' 开头的加密 token

        Returns:
            解密后的用户数据字典，失败返回 None
        """
        if not token.startswith("sk-"):
            return None
        try:
            token_body = token[3:]
            # 补充 base64 填充
            padding = 4 - (len(token_body) % 4)
            if padding != 4:
                token_body += "=" * padding
            combined = base64.b64decode(token_body)
            iv, encrypted = combined[:16], combined[16:]
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted = self._unpad(decryptor.update(encrypted) + decryptor.finalize())
            return json.loads(decrypted.decode())
        except Exception:
            return None
