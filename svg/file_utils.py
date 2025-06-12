from os import PathLike
import os
from pathlib import Path
import boto3
import botocore.exceptions as boto_exceptions
import datasets
import re
import io
import time
import logging
import threading
from PIL import Image
from botocore.config import Config
from cached_path.schemes import SchemeClient, add_scheme_client
import gcsfs
from typing import Optional, Tuple, Union, IO

from tqdm import tqdm

PathOrStr = Union[str, PathLike]
try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache

log = logging.getLogger(__name__)

def is_url(path: PathOrStr) -> bool:
    return re.match(r"[a-z0-9]+://.*", str(path)) is not None

def upload(source: PathOrStr, target: str, save_overwrite: bool = False):
    """Upload source file to a target location on GCS or S3."""
    from urllib.parse import urlparse

    source = Path(source)
    assert source.is_file()
    parsed = urlparse(target)
    if parsed.scheme == "gs":
        _gcs_upload(source, parsed.netloc, parsed.path.strip("/"), save_overwrite=save_overwrite)
    elif parsed.scheme in ("s3", "r2", "weka"):
        _s3_upload(source, parsed.scheme, parsed.netloc, parsed.path.strip("/"), save_overwrite=save_overwrite)
    else:
        raise NotImplementedError(f"Upload not implemented for '{parsed.scheme}' scheme")

def download(source: str, target: PathOrStr, save_overwrite: bool = False):
    """Download source file to a target location on GCS or S3."""
    from urllib.parse import urlparse

    target = Path(target)
    parsed = urlparse(source)
    if parsed.scheme == "gs":
        _gcs_download(parsed.netloc, parsed.path.strip("/"), target, save_overwrite=save_overwrite)
    elif parsed.scheme in ("s3", "r2", "weka"):
        _s3_download(parsed.scheme, parsed.netloc, parsed.path.strip("/"), target, save_overwrite=save_overwrite)
    else:
        raise NotImplementedError(f"Download not implemented for '{parsed.scheme}' scheme")

def open_gcs_or_local(file_path: str, mode: str = 'r') -> IO:
    """
    Open a file from Google Cloud Storage or local filesystem.

    Args:
        file_path (str): Path to the file. If the path starts with 'gs://', it is considered a GCS path.
        mode (str): Mode in which to open the file. Default is 'r' (read).

    Returns:
        IO: A file-like object.
    """
    if file_path.startswith('gs://'):
        # Process file from Google Cloud Storage
        fs = gcsfs.GCSFileSystem()
        return fs.open(file_path, mode)
    else:
        # Process local file
        return open(file_path, mode)

## GCS Utils
def get_bucket_name_from_path(gcs_path: str) -> str:
    """
    Get the bucket name from a GCS path.
    """
    return gcs_path.split("/")[2]

def get_blob_name_from_path(gcs_path: str) -> str:
    """
    Get the blob name from a GCS path.
    """
    return "/".join(gcs_path.split("/")[3:])

def upload_image_to_gcs(image: Image.Image | str, gcs_path: str, is_public=False) -> str:
    """
    Upload an image to GCS and return the image.
    """
    from google.cloud import storage
    client = storage.Client()
    bucket_name = get_bucket_name_from_path(gcs_path)
    bucket = client.get_bucket(bucket_name)

    # strip the bucket name from the path
    blob_path = get_blob_name_from_path(gcs_path) 
    blob = bucket.blob(blob_path)
    if isinstance(image, str):
        image = Image.open(image)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    blob.upload_from_string(img_byte_arr, content_type="image/jpeg")
    if is_public:
        blob.make_public()  # Make the blob publicly accessible
        return blob.public_url  # Return the public URL of the image
    else:
        return gcs_path  # Return the private URL or any other identifier

def load_image_from_gcs(image_path: str) -> Image.Image:
    from google.cloud import storage
    storage_client = storage.Client()
    bucket_name = get_bucket_name_from_path(image_path)
    bucket = storage_client.bucket(bucket_name)

    blob_path = get_blob_name_from_path(image_path)
    blob = bucket.blob(blob_path)
    image_data = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_data))
    return image

def load_dataset_from_gcs(dataset_path: str, **kwargs) -> datasets.Dataset:
    """
    Load a dataset from Google Cloud Storage.
    """
    from datasets import load_from_disk
    if dataset_path.startswith('gs://'):
            storage_options = {
                'token': os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
            }
    else:
        storage_options = None

    return load_from_disk(dataset_path, storage_options=storage_options, **kwargs)

def _gcs_download(bucket_name: str, key: str, target: Path, save_overwrite: bool = False):
    from google.cloud import storage as gcs

    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=key)
    
    for blob in blobs:
        relative_path = Path(blob.name).relative_to(key)
        target_path = target / relative_path
        if not save_overwrite and target_path.exists():
            raise FileExistsError(f"{target_path} already exists. Use save_overwrite to overwrite it.")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target_path)

def _s3_download(scheme: str, bucket_name: str, key: str, target: Path, save_overwrite: bool = False):
    err: Optional[Exception] = None
    if not save_overwrite and target.exists():
        raise FileExistsError(f"{target} already exists. Use save_overwrite to overwrite it.")
    for attempt in range(1, 4):
        try:
            _get_s3_client(scheme).download_file(bucket_name, key, str(target))
            return
        except boto_exceptions.ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                raise FileNotFoundError(f"{scheme}://{bucket_name}/{key}") from e
            err = e
        except (boto_exceptions.HTTPClientError, boto_exceptions.ConnectionError) as e:
            # ResponseStreamingError (subclass of HTTPClientError) can happen as
            # a result of a failed read from the stream (http.client.IncompleteRead).
            # Retrying can help in this case.
            err = e

        if attempt < 3:
            log.warning("%s failed attempt %d with retriable error: %s", _s3_download.__name__, attempt, err)
            _wait_before_retry(attempt)

    raise RuntimeError(f"Failed to download from {scheme}") from err

def upload_file_or_folder(local_path, bucket_name, s3_path='', endpoint_url="https://weka-aus.beaker.org:9000"):
    """
    Uploads a file or folder to an S3 bucket with a progress bar.

    :param local_path: The local file or folder path to upload.
    :param bucket_name: The name of the S3 bucket.
    :param s3_path: The S3 path prefix where files will be uploaded.
    :param endpoint_url: The S3 endpoint URL.
    """
    # Get AWS credentials from environment variables
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if not aws_access_key_id or not aws_secret_access_key:
        raise Exception("AWS credentials not found in environment variables.")

    # Create S3 client with custom endpoint
    s3_client = boto3.client('s3',
                             aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key,
                             endpoint_url=endpoint_url)

    # Define a callback class for progress bar
    class ProgressPercentage(object):
        def __init__(self, filename):
            self._filename = filename
            self._size = float(os.path.getsize(filename))
            self._seen_so_far = 0
            self._lock = threading.Lock()
            self.pbar = tqdm(total=self._size, unit='B', unit_scale=True, desc=os.path.basename(filename))

        def __call__(self, bytes_amount):
            with self._lock:
                self._seen_so_far += bytes_amount
                self.pbar.update(bytes_amount)
                if self._seen_so_far >= self._size:
                    self.pbar.close()

    if os.path.isfile(local_path):
        # Upload a single file
        filename = os.path.basename(local_path)
        s3_key = os.path.join(s3_path, filename).replace("\\", "/")
        print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
        s3_client.upload_file(local_path, bucket_name, s3_key, Callback=ProgressPercentage(local_path))
    else:
        # Upload a folder
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                s3_key = os.path.join(s3_path, relative_path).replace("\\", "/")
                print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
                s3_client.upload_file(local_file_path, bucket_name, s3_key, Callback=ProgressPercentage(local_file_path))

def get_bytes_range(source: PathOrStr, bytes_start: int, num_bytes: int) -> bytes:
    if is_url(source):
        from urllib.parse import urlparse

        parsed = urlparse(str(source))
        if parsed.scheme == "gs":
            return _gcs_get_bytes_range(parsed.netloc, parsed.path.strip("/"), bytes_start, num_bytes)
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_get_bytes_range(
                parsed.scheme, parsed.netloc, parsed.path.strip("/"), bytes_start, num_bytes
            )
        elif parsed.scheme in ("http", "https"):
            return _http_get_bytes_range(
                parsed.scheme, parsed.netloc, parsed.path.strip("/"), bytes_start, num_bytes
            )
        elif parsed.scheme == "file":
            return get_bytes_range(str(source).replace("file://", "", 1), bytes_start, num_bytes)
        else:
            raise NotImplementedError(f"get bytes range not implemented for '{parsed.scheme}' files")
    else:
        with open(source, "rb") as f:
            f.seek(bytes_start)
            return f.read(num_bytes)


def find_latest_checkpoint(dir: PathOrStr) -> Optional[PathOrStr]:
    if is_url(dir):
        from urllib.parse import urlparse

        parsed = urlparse(str(dir))
        if parsed.scheme == "gs":
            raise NotImplementedError
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_find_latest_checkpoint(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme == "file":
            return find_latest_checkpoint(str(dir).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"find_latest_checkpoint not implemented for '{parsed.scheme}' files")
    else:
        latest_step = 0
        latest_checkpoint: Optional[Path] = None
        for path in Path(dir).glob("step*"):
            if path.is_dir():
                try:
                    step = int(path.name.replace("step", "").replace("-unsharded", ""))
                except ValueError:
                    continue
                # We prioritize sharded checkpoints over unsharded checkpoints.
                if step > latest_step or (step == latest_step and not path.name.endswith("-unsharded")):
                    latest_step = step
                    latest_checkpoint = path
        return latest_checkpoint


def _gcs_upload(source: Path, bucket_name: str, key: str, save_overwrite: bool = False):
    from google.cloud import storage as gcs

    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    if not save_overwrite and blob.exists():
        raise FileExistsError(f"gs://{bucket_name}/{key} already exists. Use save_overwrite to overwrite it.")
    blob.upload_from_filename(source)


def _gcs_file_size(bucket_name: str, key: str) -> int:
    from google.api_core.exceptions import NotFound
    from google.cloud import storage as gcs

    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    try:
        blob.reload()
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket_name}/{key}")
    assert blob.size is not None
    return blob.size


def _gcs_get_bytes_range(bucket_name: str, key: str, bytes_start: int, num_bytes: int) -> bytes:
    from google.api_core.exceptions import NotFound
    from google.cloud import storage as gcs

    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    try:
        blob.reload()
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket_name}/{key}")
    return blob.download_as_bytes(start=bytes_start, end=bytes_start + num_bytes - 1)


def _get_s3_profile_name(scheme: str) -> Optional[str]:
    if scheme == "s3":
        # For backwards compatibility, we assume S3 uses the default profile if S3_PROFILE is not set.
        return os.environ.get("S3_PROFILE")
    if scheme == "r2":
        profile_name = os.environ.get("R2_PROFILE")
        if profile_name is None:
            raise RuntimeError(
                "R2 profile name is not set. Did you forget to set the 'R2_PROFILE' env var?"
            )

        return profile_name
    if scheme == "weka":
        profile_name = os.environ.get("WEKA_PROFILE")
        if profile_name is None:
            raise RuntimeError(
                "Weka profile name is not set. Did you forget to set the 'WEKA_PROFILE' env var?"
            )

        return profile_name

    raise NotImplementedError(f"Cannot get profile name for scheme {scheme}")


def _get_s3_endpoint_url(scheme: str) -> Optional[str]:
    if scheme == "s3":
        return None
    if scheme == "r2":
        r2_endpoint_url = os.environ.get("R2_ENDPOINT_URL")
        if r2_endpoint_url is None:
            raise RuntimeError(
                "R2 endpoint url is not set. Did you forget to set the 'R2_ENDPOINT_URL' env var?"
            )

        return r2_endpoint_url
    if scheme == "weka":
        weka_endpoint_url = os.environ.get("WEKA_ENDPOINT_URL")
        if weka_endpoint_url is None:
            raise RuntimeError(
                "Weka endpoint url is not set. Did you forget to set the 'WEKA_ENDPOINT_URL' env var?"
            )

        return weka_endpoint_url

    raise NotImplementedError(f"Cannot get endpoint url for scheme {scheme}")


@cache
def _get_s3_client(scheme: str):
    session = boto3.Session(profile_name=_get_s3_profile_name(scheme))
    return session.client(
        "s3",
        endpoint_url=_get_s3_endpoint_url(scheme),
        config=Config(retries={"max_attempts": 10, "mode": "standard"}),
        use_ssl=not int(os.environ.get("OLMO_NO_SSL", "0")),
    )


def _wait_before_retry(attempt: int):
    time.sleep(min(0.5 * 2**attempt, 3.0))


def _s3_upload(
    source: Path, scheme: str, bucket_name: str, key: str, save_overwrite: bool = False, max_attempts: int = 3
):
    err: Optional[Exception] = None
    if not save_overwrite:
        for attempt in range(1, max_attempts + 1):
            try:
                _get_s3_client(scheme).head_object(Bucket=bucket_name, Key=key)
                raise FileExistsError(
                    f"s3://{bucket_name}/{key} already exists. Use save_overwrite to overwrite it."
                )
            except boto_exceptions.ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    err = None
                    break
                err = e

            if attempt < max_attempts:
                log.warning("%s failed attempt %d with retriable error: %s", _s3_upload.__name__, attempt, err)
                _wait_before_retry(attempt)

        if err is not None:
            raise RuntimeError(f"Failed to check object existence during {scheme} upload") from err

    try:
        _get_s3_client(scheme).upload_file(source, bucket_name, key)
    except boto_exceptions.ClientError as e:
        raise RuntimeError(f"Failed to upload to {scheme}") from e


def _s3_file_size(scheme: str, bucket_name: str, key: str, max_attempts: int = 3) -> int:
    err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _get_s3_client(scheme).head_object(Bucket=bucket_name, Key=key)["ContentLength"]
        except boto_exceptions.ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                raise FileNotFoundError(f"s3://{bucket_name}/{key}") from e
            err = e

        if attempt < max_attempts:
            log.warning("%s failed attempt %d with retriable error: %s", _s3_file_size.__name__, attempt, err)
            _wait_before_retry(attempt)

    raise RuntimeError(f"Failed to get {scheme} file size") from err


def _s3_get_bytes_range(
    scheme: str, bucket_name: str, key: str, bytes_start: int, num_bytes: int, max_attempts: int = 3
) -> bytes:
    err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return (
                _get_s3_client(scheme)
                .get_object(
                    Bucket=bucket_name, Key=key, Range=f"bytes={bytes_start}-{bytes_start + num_bytes - 1}"
                )["Body"]
                .read()
            )
        except boto_exceptions.ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                raise FileNotFoundError(f"{scheme}://{bucket_name}/{key}") from e
            err = e
        except (boto_exceptions.HTTPClientError, boto_exceptions.ConnectionError) as e:
            # ResponseStreamingError (subclass of HTTPClientError) can happen as
            # a result of a failed read from the stream (http.client.IncompleteRead).
            # Retrying can help in this case.
            err = e

        if attempt < max_attempts:
            log.warning(
                "%s failed attempt %d with retriable error: %s", _s3_get_bytes_range.__name__, attempt, err
            )
            _wait_before_retry(attempt)

    # When torch's DataLoader intercepts exceptions, it may try to re-raise them
    # by recalling their constructor with a single message arg. Torch has some
    # logic to deal with the absence of a single-parameter constructor, but it
    # doesn't gracefully handle other possible failures in calling such a constructor
    # This can cause an irrelevant exception (e.g. KeyError: 'error'), resulting
    # in us losing the true exception info. To avoid this, we change the exception
    # to a type that has a single-parameter constructor.
    raise RuntimeError(f"Failed to get bytes range from {scheme}") from err


def _s3_find_latest_checkpoint(scheme: str, bucket_name: str, prefix: str) -> Optional[str]:
    if not prefix.endswith("/"):
        prefix = f"{prefix}/"
    response = _get_s3_client(scheme).list_objects(Bucket=bucket_name, Prefix=prefix, Delimiter="/")
    assert not response["IsTruncated"]  # need to handle this if it happens
    latest_step = 0
    latest_checkpoint: Optional[str] = None
    for item in response.get("CommonPrefixes", []):
        prefix = item["Prefix"].strip("/")
        checkpoint_name = os.path.split(prefix)[-1]
        if not checkpoint_name.startswith("step"):
            continue
        try:
            step = int(checkpoint_name.replace("step", "").replace("-unsharded", ""))
        except ValueError:
            continue
        # Make sure the checkpoint dir contains a config, otherwise the checkpoint is incomplete
        # (upload might have have failed part way through).
        try:
            _s3_file_size(scheme, bucket_name, f"{prefix}/config.yaml")
        except FileNotFoundError:
            continue
        # We prioritize sharded checkpoints over unsharded ones.
        if step > latest_step or (step == latest_step and not checkpoint_name.endswith("-unsharded")):
            latest_step = step
            latest_checkpoint = f"{scheme}://{bucket_name}/{prefix}"
    return latest_checkpoint


def _http_file_size(scheme: str, host_name: str, path: str) -> int:
    import requests

    response = requests.head(f"{scheme}://{host_name}/{path}", allow_redirects=True)
    return int(response.headers.get("content-length"))


def _http_get_bytes_range(scheme: str, host_name: str, path: str, bytes_start: int, num_bytes: int) -> bytes:
    import requests

    response = requests.get(
        f"{scheme}://{host_name}/{path}", headers={"Range": f"bytes={bytes_start}-{bytes_start+num_bytes-1}"}
    )
    result = response.content
    assert (
        len(result) == num_bytes
    ), f"expected {num_bytes} bytes, got {len(result)}"  # Some web servers silently ignore range requests and send everything
    return result


def save_hf_dataset_to_disk(
    dataset: datasets.DatasetDict | datasets.Dataset,
    hf_path: str,
    name: Optional[str],
    split: str,
    datasets_dir: PathOrStr,
):
    """
    Saves a HF dataset to disk under the `datasets_dir`. It can be used to add a HF dataset
    to `olmo_data` as follows:

    ```
    import datasets

    from olmo.util import save_hf_dataset_to_disk

    path, name, split = ...

    dataset = datasets.load_dataset(path, name=name, split=split)
    save_hf_dataset_to_disk(dataset, path, name, split, "olmo_data/hf_datasets")
    ```
    """
    dataset_path = Path(datasets_dir) / hf_path / (name or "none") / split
    return dataset.save_to_disk(str(dataset_path))


def default_thread_count() -> int:
    return int(os.environ.get("OLMO_NUM_THREADS") or min(32, (os.cpu_count() or 1) + 4))


def pass_through_fn(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def add_cached_path_clients():
    add_scheme_client(WekaClient)


class WekaClient(SchemeClient):
    recoverable_errors = SchemeClient.recoverable_errors + (
        boto_exceptions.HTTPClientError,
        boto_exceptions.ConnectionError,
    )

    scheme = "weka"

    def __init__(self, resource: str) -> None:
        SchemeClient.__init__(self, resource)
        self.bucket_name, self.path = WekaClient._split_cloud_path(resource, "weka")
        self.s3 = _get_s3_client("weka")
        self.object_info = None

    @staticmethod
    def _split_cloud_path(url: str, provider: str) -> Tuple[str, str]:
        """Split a full s3 path into the bucket name and path."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if not parsed.netloc or not parsed.path:
            raise ValueError("bad {} path {}".format(provider, url))
        bucket_name = parsed.netloc
        provider_path = parsed.path
        # Remove '/' at beginning of path.
        if provider_path.startswith("/"):
            provider_path = provider_path[1:]
        return bucket_name, provider_path

    def _ensure_object_info(self):
        if self.object_info is None:
            try:
                self.object_info = self.s3.head_object(Bucket=self.bucket_name, Key=self.path)
            except boto_exceptions.ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    raise FileNotFoundError(f"weka://{self.bucket_name}/{self.path}") from e
                raise e

    def get_etag(self) -> Optional[str]:
        self._ensure_object_info()
        assert self.object_info is not None
        return self.object_info.get("ETag")

    def get_size(self) -> Optional[int]:
        self._ensure_object_info()
        assert self.object_info is not None
        return self.object_info.get("ContentLength")

    def get_resource(self, temp_file: io.BufferedWriter) -> None:
        self.s3.download_fileobj(Fileobj=temp_file, Bucket=self.bucket_name, Key=self.path)

    def get_bytes_range(self, index: int, length: int) -> bytes:
        response = self.s3.get_object(
            Bucket=self.bucket_name, Key=self.path, Range=f"bytes={index}-{index+length-1}"
        )
        return response["Body"].read()