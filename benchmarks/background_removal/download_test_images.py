#!/usr/bin/env python3
import argparse
import json
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


def _download_image(url: str) -> Image.Image:
    req = Request(url, headers={"User-Agent": "TRELLIS.2-background-benchmark/1.0"})
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    image = Image.open(BytesIO(data))
    return image.convert("RGB")


def _shrink_long_edge(image: Image.Image, max_long_edge: int) -> Image.Image:
    long_edge = max(image.size)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    return resized


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark images with pinned URLs and metadata.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/background_removal/image_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/background_removal/images"),
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=1280,
        help="Downscale downloaded images so committed/working copies stay small.",
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    rows = payload["external_images"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        image = _download_image(row["image_url"])
        image = _shrink_long_edge(image, args.max_long_edge)
        output_path = args.output_dir / row["filename"]
        image.save(output_path, quality=92)
        print(f"saved {row['id']} -> {output_path} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
