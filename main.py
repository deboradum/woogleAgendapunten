import argparse
import requests
import urllib
import json
import uuid
import os
import re

from bs4 import BeautifulSoup


def download_vergadering(url, filepath):
    # r = requests.get(url)
    # For some reason request.get gave me 500, urllib works fine.
    r = urllib.request.urlopen(url)
    # print(r.status_code, r.reason)
    bsObj = BeautifulSoup(r, "html.parser")
    download_url = bsObj.find(href=re.compile("download"))

    if not download_url:
        print("Could not find a download URL for", url)
        raise Exception("Could not find download URL for", url)
    download_url = download_url.get("href")

    r = requests.get(download_url, stream=True)
    print(f"Downloading {download_url}")
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def parse_urls_from_file(filepath):
    with open(filepath, "r"):
        for i, line in enumerate(filepath):
            url = line.replace("\n", "").strip()
            handle_url(url)


def transcribe_mlx(video_path, output_path):
    import mlx_whisper
    result = mlx_whisper.transcribe(
        video_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        verbose=None,
        **{"language": "nl", "task": "transcribe"},
    )
    with open(output_path, "w") as f:
        json.dump(result, f)


def transcribe_torch(video_path, output_path):

    result = None
    with open(output_path, "w") as f:
        json.dump(result, f)


def transcribe(video_path, output_path, use_mlx):
    if use_mlx:
        transcribe_mlx(video_path, output_path)
    else:
        transcribe_torch(video_path, output_path)


def parse_transcription(transcription_path, output_path):
    return


def handle_url(url):
    video_path = f"{url}.mp4"
    transcription_path = f"{url}_transcript.json"
    final_path = f"{url}_final.json"

    download_vergadering(url, video_path)
    transcribe(video_path, transcription_path, args.mlx)
    os.remove(video_path)
    parse_transcription(transcription_path, final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=None, help="Url of the page")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Filepath containing multiple urls, one url per line",
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Use Apple MLX based Whisper, if not passed uses torch based Whisper",
    )
    args = parser.parse_args()

    if args.url is None and args.file is None:
        print("Either url or file should be passed as an argument!")
        exit()

    if args.file:
        if args.file.endswith(".txt"):
            print("Please provide a txt file")
            exit()
        print("Retrieving information for file:", args.file)
        parse_urls_from_file(args.file)

    if args.url:
        print("Retrieving information for url:", args.url)
        handle_url(args.url)
