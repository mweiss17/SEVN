import requests
import zipfile
import os
from tqdm import tqdm
import argparse
from SEVN_gym.data import DATA_PATH


ZENODO_URL = 'https://zenodo.org/record/3526490/files/'


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


def download_metadata():
    # Download and unzip metadata 
    print('Downloading metadata')
    if os.path.isfile(DATA_PATH + 'coord.hdf5'):
        print("Already Downloaded coord.hdf5")
    else:
        download_file(ZENODO_URL + "coord.hdf5", DATA_PATH + 'coord.hdf5')
    if os.path.isfile(DATA_PATH + 'graph.pkl'):
        print("Already Downloaded graph.pkl")
    else:
        download_file(ZENODO_URL + "graph.pkl", DATA_PATH + 'graph.pkl')
    if os.path.isfile(DATA_PATH + 'label.hdf5'):
        print("Already Downloaded label.hdf5")
    else:
        download_file(ZENODO_URL + "label.hdf5", DATA_PATH + 'label.hdf5')
    print('Metadata Download Finished')


def download_low_res():
    print('Downloading images')
    if os.path.isfile(DATA_PATH + 'images.hdf5'):
        print("Already Downloaded images.hdf5")
    else:
        download_file(ZENODO_URL + "images.hdf5", DATA_PATH + 'images.hdf5')
    print('Download finished')


def download_high_res():
    print('Downloading images')
    if os.path.isfile(DATA_PATH + 'high-res-panos.zip'):
        print("Already Downloaded high-res-panos.zip")
    else:
        download_file(ZENODO_URL + "high-res-panos.zip", DATA_PATH + 'high-res-panos.zip')
    print('Download finished')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--high-res', action="store_true",
                        help='Download high-resolution images')
    args = parser.parse_args()
    download_metadata()
    if args.high_res:
        download_high_res()
    else:
        download_low_res()


if __name__ == '__main__':
    main()