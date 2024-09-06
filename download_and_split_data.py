import csv
import os
import random
import tarfile
from typing import List, Tuple
import requests

DATASET_URL: str = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

def download_and_unpack_tar_gz(url: str, extract_path='.'):
    """
    Downloads a tar.gz file from a specified URL and unpacks it to a given directory.

    :param url: URL of the tar.gz file to download
    :param extract_path: Directory to extract the contents to
    """
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open a file to write the content to
        with open('data.tar.gz', 'wb') as f:
            f.write(response.raw.read())
        
        # Open the tar.gz file
        with tarfile.open('data.tar.gz', 'r:gz') as tar:
            # Extract its contents
            tar.extractall(path=extract_path)
            print("Files extracted successfully.")
            os.remove('data.tar.gz')
    else:
        raise RuntimeError(f"Failed to download file. Status code: {response.status_code}")

def extract_reviews(base_path: str = ".", split: str = "train", total_reviews: int = 1750, pos_proportion: float = 0.5) -> List[Tuple[str, int]]:
    """
    Crawls directories for positive and negative reviews and returns a list of tuples (review text, label).

    :param base_path: The base directory where 'aclImdb' folder is located.
    :return: a list of reviews and their scores
    """
    pos_reviews: List[str] = []
    neg_reviews: List[str] = []
    num_positive: int = int(pos_proportion * total_reviews)
    num_negative: int = int((1 - pos_proportion) * total_reviews)


    # Define paths to the positive and negative review directories
    pos_path: str = os.path.join(base_path, 'aclImdb', split, 'pos')
    neg_path: str = os.path.join(base_path, 'aclImdb', split, 'neg')

    # Read all files in the positive reviews directory
    for filename in os.listdir(pos_path):
        with open(os.path.join(pos_path, filename), 'r', encoding='utf-8') as file:
            pos_reviews.append(file.read())
            if len(pos_reviews) >= num_positive:
                break

    # Read all files in the negative reviews directory
    for filename in os.listdir(neg_path):
        with open(os.path.join(neg_path, filename), 'r', encoding='utf-8') as file:
            neg_reviews.append(file.read())
            if len(neg_reviews) >= num_negative:
                break

    result: List[Tuple[str, int]] = []
    assert num_positive <= len(pos_reviews), f"unable to extract {num_positive} positive reviews"
    assert num_negative <= len(neg_reviews), f"unable to extract {num_negative} negative reviews"

    # add the reviews and return
    result.extend([(review, 1) for review in pos_reviews[:num_positive]])
    result.extend([(review, 0) for review in neg_reviews[:num_negative]])
    return result

if __name__ == '__main__':
    if not os.path.exists('aclImdb'):
        download_and_unpack_tar_gz(DATASET_URL)
    train_reviews = extract_reviews(split="train", pos_proportion=0.5, total_reviews=1650)

    # split train into train (1400) and dev (250):
    random.seed(42)
    random.shuffle(train_reviews)
    train_reviews, dev_reviews = train_reviews[:-250], train_reviews[-250:]

    print("positive reviews in train: ", sum(label for (review, label) in train_reviews))
    print("positive reviews in dev: ", sum(label for (review, label) in dev_reviews))

    # get test set
    test_reviews = extract_reviews(split="test", pos_proportion=0.5, total_reviews=250)

    # write all out as csvs:
    for split_name, split_data in zip(
        ["train", "dev", "test"],
        [train_reviews, dev_reviews, test_reviews]
        ):
        print(f"writing {split_name}.csv")
        with open(f"data/{split_name}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label'])
            writer.writerows(split_data)