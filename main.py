import argparse
import requests
import urllib
import json
import os
import re

from bs4 import BeautifulSoup


def download_vergadering(url, filepath):
    if os.path.isfile(filepath):
        print("Vergadering already downlaoded.")
        return

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
    print(f"Downloaded {download_url}")


def get_agenda(url):
    print("Retrieving agenda")

    r = requests.get(url)
    bsObj = BeautifulSoup(r.content, "html.parser")

    agenda_json = []
    agenda_items = [li for li in bsObj.find_all("li", class_="agenda_item")]
    # print(agenda_items)
    for item in agenda_items:
        agenda_point = {}
        btn = item.find("button", class_="item_title")
        if not btn:
            print("No button found")
            continue

        # span = btn.find("span", class_="item_prefix")
        # Skip sub agenda item (for now, perhaps)
        # if not span or not span.text.strip().endswith("."):
        #     print("Sub item")
        #     continue

        agenda = btn.get_text(strip=True, separator=" ")
        agenda_point["agendaPoint"] = agenda
        time_span = item.find("span", class_="item_time")
        if time_span:
            time = time_span.text.strip().replace("tijdsduur:", "").strip()
            agenda_point["time"] = time

        agenda_json.append(agenda_point)

    print("Retrieved agenda")

    return agenda_json


def parse_urls_from_file(filepath):
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            url = line.replace("\n", "")
            print("Doing", url)
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
        json.dump(result, f, indent=4)


def transcribe_torch(video_path, output_path):
    import whisper

    # 'base', 'small', 'medium' or 'large'
    model = whisper.load_model("medium")
    result = model.transcribe(video_path, language="nl")

    result = None
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)


def transcribe(video_path, output_path, use_mlx):
    if os.path.isfile(output_path):
        print("Transcription already exists.")
        return

    print(f"Transcribing {video_path}")
    if use_mlx:
        transcribe_mlx(video_path, output_path)
    else:
        transcribe_torch(video_path, output_path)
    print(f"Transcribed {video_path}")


def time_string_to_seconds(time_string):
    hours, minutes, seconds = map(int, time_string.split(":"))
    total_seconds = hours * 3600 + minutes * 60 + seconds

    return total_seconds


def parse_transcription(transcription_path, output_path, agenda):
    if os.path.isfile(output_path):
        print("vergadering already parsed.")
        return

    print(f"Parsing {transcription_path}")
    agenda_and_text = []
    data = None
    with open(transcription_path, "r") as f:
        data = json.load(f)
    segments = data["segments"]
    running_time = 0
    for agendapunt in agenda:
        if not agendapunt.get("time"):
            continue
        start = running_time
        end = running_time + time_string_to_seconds(agendapunt["time"])
        # Get all whisper extracted text between agendapunt start and end.
        texts = [segment["text"].replace("...", "") for segment in segments if segment["start"] >= start and segment["start"] <= end]
        agenda_and_text.append(
            {
                "agendapunt": agendapunt["agendaPoint"],
                "start": start,
                "end": end,
                "text": "".join(texts),
            }
        )

        running_time += time_string_to_seconds(agendapunt["time"])

    with open(output_path, "w") as f:
        json.dump(agenda_and_text, f)

    print(f"Parsed {transcription_path}")


def handle_url(url):
    vergadering_code = url.split("/")[-1]
    video_path = f"{vergadering_code}.mp4"
    transcription_path = f"{vergadering_code}_transcript.json"
    final_path = f"{vergadering_code}_final.json"


    agenda = get_agenda(url)
    if not agenda:
        print(f"No agenda items could be found for {url}!")
        return
    # if not agenda[0].get("time"):
    #     print(f"Agenda for {url} has no timestamps!")
    #     return

    # download_vergadering(url, video_path)
    # transcribe(video_path, transcription_path, args.mlx)
    # os.remove(video_path)
    parse_transcription(transcription_path, final_path, agenda)


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
        if not args.file.endswith(".txt"):
            print("Please provide a txt file")
            exit()
        print("Retrieving information for file:", args.file)
        parse_urls_from_file(args.file)

    if args.url:
        print("Retrieving information for url:", args.url)
        handle_url(args.url)
