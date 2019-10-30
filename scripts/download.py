import requests
import zipfile
from tqdm import tqdm
import argparse
from SEVN_gym.data import DATA_PATH


METADATA_URL = 'https://zenodo.org/record/3521988/files/SEVN-metadata.zip?download=1'
LOW_RES_PANOS_URL = 'https://zenodo.org/record/3521905/files/images.hdf5?download=1'
HIGH_RES_PANOS_URL = "TODO"


def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)


def download_low_res():
    # Download and unzip metadata 
    print('Downloading metadata')
    download_file(METADATA_URL, DATA_PATH + 'SEVN-metadata.zip')

    with zipfile.ZipFile(DATA_PATH + 'SEVN-metadata.zip', 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH)

    # Download images
    print('Downloading images')
    download_file(LOW_RES_PANOS_URL, DATA_PATH + 'images.hdf5')
    print('Download finished')


def download_high_res():
    print('High resolution dataset is not yet available')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--high-res', action="store_true",
                        help='Download high-resolution images')
    args = parser.parse_args()
    if args.high_res:
        download_high_res()
    else:
        download_low_res()


if __name__ == '__main__':
    main()