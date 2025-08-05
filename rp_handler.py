import os
import traceback
import requests
import boto3
import runpod
from playdiffusion import PlayDiffusion, RVCInput
from botocore.exceptions import ClientError, ConnectionClosedError
from botocore.config import Config
from uuid import uuid4
import io
import mimetypes
import numpy as np
from scipy.io import wavfile
from urllib.parse import urlparse
import tempfile
import torch


def upload_to_s3(audio_data: bytes, bucket_name: str = None, object_key_prefix: str = "voicetovoice", file_extension: str = ".wav") -> str:
    if not audio_data:
        raise ValueError("audio_data is required")

    aws_access_key_id = os.environ.get(
        "AWS_ACCESS_KEY_ID", "DO801QRYN7XNMKV79HBC")
    aws_secret_access_key = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", "inKxzsLVWYaxS3kY4R5i9MvwMRw/0h3Ym7CeHV8T6U4")
    endpoint_url = os.environ.get(
        "SPACES_ENDPOINT_URL", "https://sfo3.digitaloceanspaces.com")

    if not bucket_name:
        print("Warning: bucket_name not provided. Using default bucket 'denoise'.")
        bucket_name = "denoise"

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
        config=Config(retries={'max_attempts': 3, 'mode': 'standard'})
    )

    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"Bucket '{bucket_name}' does not exist. Creating bucket...")
            try:
                s3_client.create_bucket(Bucket=bucket_name)
                print(f"Created bucket '{bucket_name}'.")
            except ClientError as ce:
                raise ClientError(
                    f"Failed to create bucket '{bucket_name}': {str(ce)}", operation_name="create_bucket")
        else:
            raise ClientError(
                f"Error checking bucket '{bucket_name}': {str(e)}", operation_name="head_bucket")

    content_type = mimetypes.guess_type(f"file{file_extension}")[
        0] or f"audio/{file_extension.lstrip('.')}"
    object_key = f"{object_key_prefix}/{uuid4()}{file_extension}" if object_key_prefix else f"{uuid4()}{file_extension}"

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=audio_data,
            ContentType=content_type,
            ACL='public-read'
        )
        parsed_endpoint = urlparse(endpoint_url)
        base_domain = parsed_endpoint.netloc
        spaces_url = f"https://{bucket_name}.{base_domain}/{object_key}"
        print(
            f"Uploaded file to s3://{bucket_name}/{object_key}, accessible at {spaces_url}")
        return spaces_url
    except ClientError as e:
        raise ClientError(
            f"Failed to upload to Spaces: {str(e)}", operation_name="put_object")
    except ConnectionClosedError as e:
        raise RuntimeError(f"Connection error during S3 upload: {str(e)}")


def voice_conversion(
    source_audio_path: str,
    target_audio_path: str,
    bucket_name: str = None,
    object_key_prefix: str = ""
) -> str:
    if not os.path.exists(source_audio_path):
        raise FileNotFoundError(f"Source audio not found: {source_audio_path}")
    if not os.path.exists(target_audio_path):
        raise FileNotFoundError(f"Target voice not found: {target_audio_path}")

    print(f"Starting voice conversion using PlayDiffusion.rvc")

    rvc_model = PlayDiffusion()
    rvc_input = RVCInput(
        source_speech=source_audio_path,
        target_voice=target_audio_path
    )

    try:
        output_frequency, output_audio = rvc_model.rvc(rvc_input)

        if not isinstance(output_audio, np.ndarray) or output_audio.size == 0:
            raise RuntimeError(
                "Voice conversion failed or returned empty audio")

        if output_audio.ndim == 1:
            output_audio = output_audio.reshape(-1, 1)

        with io.BytesIO() as wav_buffer:
            wavfile.write(wav_buffer, output_frequency, output_audio)
            wav_bytes = wav_buffer.getvalue()

        spaces_url = upload_to_s3(
            audio_data=wav_bytes,
            bucket_name=bucket_name,
            object_key_prefix=object_key_prefix,
            file_extension=".wav"
        )
        return spaces_url

    except Exception as e:
        raise RuntimeError(
            f"Voice conversion failed: {str(e)}\nTraceback: {traceback.format_exc()}"
        )


def handler(event):
    print(f"Worker Start")
    try:
        input_data = event.get('input', {})
        source_audio_url = input_data.get('source_audio_url')
        target_audio_url = input_data.get('target_audio_url')
        bucket_name = input_data.get('bucket_name')
        object_key_prefix = input_data.get('object_key_prefix', "")

        if not source_audio_url or not target_audio_url:
            raise ValueError(
                "Both source_audio_url and target_audio_url are required")

        # Download source audio
        response = requests.get(source_audio_url, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download source audio: {response.status_code}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as src_temp:
            src_temp.write(response.content)
            src_path = src_temp.name

        # Download target audio
        response = requests.get(target_audio_url, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download target audio: {response.status_code}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tgt_temp:
            tgt_temp.write(response.content)
            tgt_path = tgt_temp.name

        print(f"Downloaded source audio to: {src_path}")
        print(f"Downloaded target voice to: {tgt_path}")

        spaces_url = voice_conversion(
            source_audio_path=src_path,
            target_audio_path=tgt_path,
            bucket_name=bucket_name,
            object_key_prefix=object_key_prefix
        )

        return {
            'status': 'success',
            'audio_url': spaces_url
        }

    except (FileNotFoundError, ValueError, RuntimeError, ClientError) as e:
        print(f"Error: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }
    except Exception as e:
        print(
            f"Unexpected error: {str(e)}\nTraceback: {traceback.format_exc()}")
        return {
            'status': 'error',
            'message': f"Unexpected error: {str(e)}\nTraceback: {traceback.format_exc()}"
        }
    finally:
        for path in ['src_path', 'tgt_path']:
            if path in locals() and os.path.exists(locals()[path]):
                try:
                    os.unlink(locals()[path])
                    print(f"Cleaned up temporary file: {locals()[path]}")
                except Exception as e:
                    print(
                        f"Failed to clean up file {locals()[path]}: {str(e)}")


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
