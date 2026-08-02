"""Detects the real file format from content, not the (frequently misleading) file extension.

Confirmed empirically: dataset/media/images/*.jpg and dataset/media/audio/*.mp3 are a mix of
JPEG/PNG/WebP/AVIF and MP3/WAV/M4A respectively, all using the wrong extension.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

_PIL_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "AVIF": "image/avif",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}


def sniff_image_mime(path: Path) -> str:
    with Image.open(path) as im:
        fmt = im.format
    if fmt not in _PIL_FORMAT_TO_MIME:
        raise ValueError(f"Unrecognized image format {fmt!r} for {path}")
    return _PIL_FORMAT_TO_MIME[fmt]


def sniff_audio_mime(path: Path) -> str:
    with open(path, "rb") as f:
        header = f.read(16)

    if header[:3] == b"ID3" or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        return "audio/mp3"
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "audio/wav"
    if header[4:8] == b"ftyp":
        # Confirmed via live API test: both true M4A (M4A brand) and generic ISO-media MP4
        # audio containers (isom/mp42 brand) are accepted by Gemini as mime_type audio/m4a.
        return "audio/m4a"

    raise ValueError(f"Unrecognized audio format for {path} (header={header!r})")
