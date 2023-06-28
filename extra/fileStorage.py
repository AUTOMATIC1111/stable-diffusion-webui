import base64
import uuid
from io import BytesIO
from minio import Minio, S3Error

from extra.loadYamlFile import ExtraConfig


class ExtraFileStorage:
    def __init__(self):
        config = ExtraConfig()
        self.config_data = config.get_config()
        # 初始化minio客户端
        self.client = Minio(self.config_data['upload']['server-addr'],
                            access_key=self.config_data['upload']['access_key'],
                            secret_key=self.config_data['upload']['secret_key'],
                            secure=False)

    def saveBase642Minio(self, base64_data):
        # 解码base64数据为二进制数据
        binary_data = base64.b64decode(base64_data)

        # 生成不重复的文件名
        filename = str(uuid.uuid4()) + ".png"

        try:
            # 将二进制数据上传到MinIO
            self.client.put_object(
                self.config_data['upload']['bucket_name'],
                filename,
                BytesIO(binary_data),
                len(binary_data)
            )

            # 返回访问路径
            return f"{self.config_data['upload']['bucket_name']}/{filename}"
        except S3Error as err:
            print(err)
            return None

    def saveBase64Files(self, images: []):
        rs = []
        if images:
            for img in images:
                url = self.saveBase642Minio(img)
                rs.append(url)

        return rs

    def saveByte2Minio(self, byte_data, file_extension):
        # 生成不重复的文件名
        filename = str(uuid.uuid4()) + '.' + file_extension

        try:
            # 将二进制数据上传到MinIO
            self.client.put_object(
                self.config_data['upload']['bucket_name'],
                filename,
                BytesIO(byte_data),
                len(byte_data)
            )

            # 返回访问路径
            return f"{self.config_data['upload']['bucket_name']}/{filename}"
        except S3Error as err:
            print(err)
            return None
