from pathlib import Path

import requests
from tqdm import tqdm


BASE_URL = "https://object.pouta.csc.fi/OPUS-Europarl/v8/xml"

FILES = [
    "en.zip",
    "de.zip",
    "fr.zip",
    "de-en.xml.gz",
    "en-fr.xml.gz",
]

OUTPUT_DIR = Path("data/raw/europarl")
CHUNK_SIZE = 1024 * 1024


def download_file(filename):
    url = f"{BASE_URL}/{filename}"
    destination = OUTPUT_DIR / filename
    temporary = OUTPUT_DIR / f"{filename}.part"

    if destination.exists():
        print(f"Already exists, skipping: {destination}")
        return

    print(f"\nDownloading: {url}")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with temporary.open("wb") as output_file:
            progress = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=filename,
            )

            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    output_file.write(chunk)
                    progress.update(len(chunk))

            progress.close()

    if total_size and temporary.stat().st_size != total_size:
        raise RuntimeError(
            f"Incomplete download for {filename}: "
            f"expected {total_size}, received {temporary.stat().st_size}"
        )

    temporary.replace(destination)
    print(f"Saved: {destination}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        download_file(filename)

    print("\nEuroparl downloads completed.")


if __name__ == "__main__":
    main()