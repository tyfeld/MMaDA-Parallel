from contextlib import contextmanager
from io import BytesIO
import logging
import os
import time
from typing import Union

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)


@contextmanager
def no_proxy():
    d_original = {}
    for variable in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        d_original[variable] = os.environ.pop(variable, None)

    try:
        # operations within the context
        yield
    finally:
        for variable, original_value in d_original.items():
            if original_value is not None:
                os.environ[variable] = original_value
            else:
                os.environ.pop(variable, None)


def read_general(path) -> Union[str, BytesIO]:
    with no_proxy():
        # if "s3://" in path:
        #     init_ceph_client_if_needed()
        #     file_bytes = BytesIO(client.get(path, update_cache=True))
        #     return file_bytes
        if "lc2" in path:
            init_ceph_client_if_needed()
            file_bytes = BytesIO(client.get(path, update_cache=True))
            return file_bytes
        else:
            return path


def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa

        client = Client("/mnt/petrelfs/xinyi/petreloss.conf")
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")


client = None
