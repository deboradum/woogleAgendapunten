# Woogle Agendapunten
A small repo to retrieve agendapunt accurate video transcripts from NotuBiz municipality meeting recordings. This repo uses Whisper to transcribe videos, and supports MacOS Mx use with Apple's MLX framework, as well as regular use with a standard torch implementation.

# Prerequisites
For Whisper to use, make sure `ffmpeg` is installed by either runnning
```
brew install ffmpeg
```
or 
```
sudo apt install ffmpeg
```
depending on your device. After, install `requirements.txt` or `requirements_mac.txt` depending on your device.

It is highly recommended to use a device that has a GPU, otherwise transcription times can take a long time due to the fact that the meetings are often multiple hours.

# Usage 
```
usage: main.py [-h] [--url URL] [--file FILE] [--mlx]

options:
  -h, --help   show this help message and exit
  --url URL    Url of the page
  --file FILE  Filepath containing multiple urls, one url per line
  --mlx        Use Apple MLX based Whisper, if not passed uses torch based Whisper
```

To use the tool, either provide a url or filepath. When providing a filepath, the txt file should contain one url per line. When running on an Apple device containing a Mx chip, also pass `--mlx`. Example usage:

```
$  python main.py --url "https://wijdemeren.notubiz.nl/vergadering/1176096" --mlx  // parse the url with Apple mlx Whisper
$  python main.py --file "vergaderingen.txt"  // parse all urls in vergaderingen.txt with Torch whisper
```
