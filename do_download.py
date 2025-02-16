import requests
from tqdm import tqdm
import random
from bs4 import BeautifulSoup
import argparse
import os

# Base URL for Project Gutenberg's plain text files
BASE_URL = "https://www.gutenberg.org/files/{}/{}-0.txt"

# URL to fetch a list of recent eBooks (we'll use this to find random book IDs)
RECENT_EBOOKS_URL = "https://www.gutenberg.org/ebooks/search/?sort_order=random"

# Folder to save downloaded books
DOWNLOAD_FOLDER = "./data/original_text"


def fetch_random_book_ids(num_books):
    """
    Fetches a list of random book IDs from Project Gutenberg's recent eBooks page.
    """
    try:
        response = requests.get(RECENT_EBOOKS_URL)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all book links (they are in <li> tags with class 'booklink')
        book_links = soup.find_all("li", class_="booklink")
        book_ids = []

        # Extract book IDs from the links
        for link in book_links:
            book_id = link.find("a")["href"].split("/")[-1]
            if book_id.isdigit():  # Ensure it's a valid numeric ID
                book_ids.append(int(book_id))
            if len(book_ids) >= num_books * 2:  # Fetch extra IDs to account for failures
                break

        return book_ids
    except Exception as e:
        print(f"Error fetching book IDs: {e}")
        return []


def download_book(book_id, filepath):
    """
    Downloads a book from Project Gutenberg and saves it as a .txt file.
    Includes a progress bar using tqdm.
    Returns True if the download was successful, False otherwise.
    """
    url = BASE_URL.format(book_id, book_id)
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        with open(filepath, "wb") as file, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))
        print(f"Downloaded: {filepath}")
        return True
    else:
        print(f"Failed to download book ID {book_id}. Status code: {response.status_code}")
        return False


def generate_filepath(index):
    """
    Generates a filepath for the book based on its index.
    Example: ./data/original_text/book_1.txt
    """
    filename = f"book_{index}.txt"
    return os.path.join(DOWNLOAD_FOLDER, filename)


def download_books(book_ids, num_books, retry_count):
    """
    Downloads multiple books from Project Gutenberg.
    If a book fails to download, it picks a different random book ID.
    Ensures exactly `num_books` are downloaded.
    """
    successful_downloads = 0
    index = 1

    while successful_downloads < num_books and book_ids:
        book_id = book_ids.pop(0)  # Get the next book ID
        filepath = generate_filepath(index)
        retries = 0

        while retries <= retry_count:
            if download_book(book_id, filepath):
                successful_downloads += 1
                index += 1
                break
            else:
                retries += 1
                if retries <= retry_count:
                    # Pick a different random book ID for the next attempt
                    if book_ids:
                        book_id = book_ids.pop(0)
                        print(f"Retrying with a different book ID: {book_id}...")
                    else:
                        print("No more book IDs to try.")
                        break
                else:
                    print(f"Max retries reached. Skipping...")

    if successful_downloads < num_books:
        print(f"Warning: Only {successful_downloads} out of {num_books} books were downloaded.")


def main():
    """
    Main function to execute the book download process.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Download random books from Project Gutenberg.")
    parser.add_argument(
        "-n", "--num_books",
        type=int,
        required=True,
        help="The number of books to download.",
    )
    parser.add_argument(
        "-r", "--retry_count",
        type=int,
        default=3,
        help="The maximum number of retries for each book if the download fails.",
    )
    args = parser.parse_args()

    # Validate the number of books and retry count
    if args.num_books <= 0:
        print("Please enter a number greater than 0 for the number of books.")
        return
    if args.retry_count < 0:
        print("Retry count must be a non-negative integer.")
        return

    # Create the download folder if it doesn't exist
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

    # Fetch random book IDs (fetch extra to account for failures)
    print("Fetching random book IDs...")
    book_ids = fetch_random_book_ids(args.num_books)
    if not book_ids:
        print("No book IDs found. Please try again later.")
        return

    print(f"Downloading {args.num_books} books to '{DOWNLOAD_FOLDER}'...")

    # Download the books
    download_books(book_ids, args.num_books, args.retry_count)
    print("Download complete!")


# Run the main function
if __name__ == "__main__":
    main()