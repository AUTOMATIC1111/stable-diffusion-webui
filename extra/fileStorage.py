import base64
import uuid
from io import BytesIO
import boto3

from extra.loadYamlFile import ExtraConfig


class ExtraFileStorage:
    def __init__(self):
        config = ExtraConfig()
        self.config_data = config.get_config()
        # 初始化客户端
        self.client = boto3.client('s3', aws_access_key_id=self.config_data['upload']['access_key'],
                                   aws_secret_access_key=self.config_data['upload']['secret_key'],
                                   region_name=self.config_data['upload']['server-addr'])

    def saveBase642Server(self, base64_data):
        # 解码base64数据为二进制数据
        binary_data = base64.b64decode(base64_data)

        # 生成不重复的文件名
        filename = str(uuid.uuid4()) + ".png"

        try:
            # 将二进制数据上传到Server
            self.client.put_object(Bucket=self.config_data['upload']['bucket_name'],
                                   Key=filename,
                                   Body=BytesIO(binary_data))

            # 返回访问路径
            return f"{self.config_data['upload']['bucket_name']}/{filename}"
        except Exception as err:
            print(err)
            return None

    def saveBase64Files(self, images: []):
        rs = []
        if images:
            for img in images:
                url = self.saveBase642Server(img)
                rs.append(url)

        return rs

    def saveByte2Server(self, byte_data, file_extension):
        # 生成不重复的文件名
        filename = str(uuid.uuid4()) + '.' + file_extension

        try:
            # 将二进制数据上传到Server
            self.client.put_object(Bucket=self.config_data['upload']['bucket_name'],
                                   Key=filename,
                                   Body=BytesIO(byte_data))

            # 返回访问路径
            return f"{self.config_data['upload']['bucket_name']}/{filename}"
        except Exception as err:
            print(err)
            return None
