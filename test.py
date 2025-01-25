import random
import string
import requests
from multiprocessing import Pool, cpu_count

def check_streamable_link(ending):
    base_url = "https://streamable.com/"
    url = base_url + ending
    # print(f"Testing: {url}")
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code == 200:
            # print(f"Working link found: {url}")
            return url
    except requests.RequestException:
        pass
    return None

def generate_random_combinations():
    characters = string.ascii_lowercase + string.digits
    while True:
        yield ''.join(random.choices(characters, k=6))

def main():
    num_workers = cpu_count()
    with Pool(num_workers) as pool:
        results = pool.imap_unordered(check_streamable_link, generate_random_combinations())
        for result in results:
            if result:
                print(f"\nFound a working link: {result}")
                break

if __name__ == "__main__":
    main()
