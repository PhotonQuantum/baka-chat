import oss2


class BaseOSS:
    def __init__(self, access_key: str, access_secret: str, bucket_name: str, internal: bool = True):
        """ An OSS object implements all methods needed for interacting with Aliyun OSS. """
        oss_url = "oss-cn-hongkong-internal.aliyuncs.com" if internal else "oss-cn-hongkong.aliyuncs.com"

        self.auth = oss2.Auth(access_key, access_secret)
        self.bucket = oss2.Bucket(self.auth, oss_url, bucket_name)
