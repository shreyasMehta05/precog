import requests

def download_dictionary(url: str, output_path: str) -> None:
    """Download the dictionary from the provided URL and save it to output_path."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded dictionary successfully to {output_path}")
    else:
        print(f"Failed to download dictionary. Status code: {response.status_code}")

if __name__ == "__main__":
    # dict_url = "https://dl.fbaipublicfiles.com/arrival/dictionaries/hi-en.txt"
    # output_file = "./data/hi_en_dictionary.txt"  # Change this to your desired output location
    dict_url = "https://dl.fbaipublicfiles.com/arrival/dictionaries/hi-en.5000-6500.txt"
    output_file = "./data/test_dictionary.txt"  # Change this to your desired output location
    download_dictionary(dict_url, output_file)
