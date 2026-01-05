import requests
import os
from urllib.parse import urlparse
import hashlib

def get_filename(url, content_type):
    """Extract filename from URL or generate one"""
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    if not filename:
        ext = content_type.split("/")[-1]
        filename = f"downloaded_image.{ext}"

    return filename


def is_duplicate(content, folder):
    """Check for duplicate images using hash"""
    file_hash = hashlib.md5(content).hexdigest()

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                if hashlib.md5(f.read()).hexdigest() == file_hash:
                    return True
    return False


def main():
    print("Welcome to the Ubuntu Image Fetcher")
    print("A tool for mindfully collecting images from the web\n")

    urls = input("Enter image URL(s) separated by commas:\n").split(",")

    os.makedirs("Fetched_Images", exist_ok=True)

    for url in urls:
        url = url.strip()

        try:
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                print(f"✗ Skipped (not an image): {url}")
                continue

            content = response.content

            # Prevent duplicates
            if is_duplicate(content, "Fetched_Images"):
                print(f"⚠ Duplicate skipped: {url}")
                continue

            filename = get_filename(url, content_type)
            filepath = os.path.join("Fetched_Images", filename)

            with open(filepath, "wb") as f:
                f.write(content)

            print(f"✓ Successfully fetched: {filename}")
            print(f"✓ Image saved to {filepath}\n")

        except requests.exceptions.RequestException as e:
            print(f"✗ Connection error for {url}: {e}")
        except Exception as e:
            print(f"✗ Unexpected error for {url}: {e}")

    print("Connection strengthened. Community enriched.")


if __name__ == "__main__":
    main()