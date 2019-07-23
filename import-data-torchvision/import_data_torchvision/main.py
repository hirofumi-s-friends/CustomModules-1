import os

import fire
import ruamel.yaml
from torchvision import datasets
from azure.storage.blob import BlockBlobService


def write_meta(dataset_name, yaml_path):
    data = {'type': 'torchvision-dataset', 'dataset-name': dataset_name}
    with open(yaml_path, 'w') as fout:
        ruamel.yaml.round_trip_dump(data, fout)


DATASET_NAMES = {
    'CIFAR10', 'CIFAR100', 'Caltech101', 'Caltech256', 'CelebA', 'FashionMNIST',
    'ImageNet', 'KMNIST', 'MNIST', 'Omniglot', 'PhotoTour',
    'SBDataset', 'SBU', 'SEMEION', 'STL10', 'SVHN', 'VOCDetection', 'VOCSegmentation',
}


def download_from_dataset(dataset_name, output_folder):
    if dataset_name not in DATASET_NAMES:
        raise Exception(f"Not a valid dataset name: {dataset_name}")
    load_data_method = getattr(datasets, dataset_name)
    print(f"Start downloading dataset {dataset_name}")
    ds = load_data_method(output_folder, download=True)
    print(f"Dataset downloaded: {ds}")
    return ds


def download_from_azure(output_folder, account_name, account_key, container, data_folder):
    args = locals()
    for key in ['account_name', 'account_key', 'container', 'data_folder']:
        if not args.get(key):
            raise ValueError(f"Input value '{key}' cannot be empty")

    # Ensure data_folder is a folder.
    if data_folder[-1] != '/':
        data_folder += '/'
    prefix_len = len(data_folder)

    def valid_image_folder_path(blob_path):
        file_name = blob_path[prefix_len:]
        if len(file_name.split('/')) != 2:
            print(f"Not a valid image folder path: {file_name}")
            return False
        _, ext = os.path.splitext(file_name)
        if ext not in datasets.folder.IMG_EXTENSIONS:
            print(f"Not a valid image extension: {_} {ext}")
            return False
        return True

    blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
    blob_paths = filter(valid_image_folder_path, blob_service.list_blob_names(container, prefix=data_folder))
    files = []
    for blob_path in blob_paths:
        file_name = blob_path[prefix_len:]
        print(f"Start downloading file: {file_name}")
        folder = os.path.dirname(os.path.join(output_folder, file_name))
        os.makedirs(folder, exist_ok=True)

        blob_service.get_blob_to_path(container_name=container, blob_name=blob_path,
                                      file_path=os.path.join(output_folder, file_name))
        files.append(file_name)
        print(f"End downloading file: {file_name}")

    if len(files) == 0:
        raise ValueError(f"Data folder '{data_folder}' doesn't contain any file")
    print(f"Start constructing dataset.")
    ds = datasets.ImageFolder(output_folder)
    print(f"Dataset constructed: {ds}")
    return ds


def download_dataset(dataset_name, output_folder, account_name="", account_key="", container="", data_folder=""):
    if dataset_name == 'ImageFolder':
        download_from_azure(output_folder, account_name, account_key, container, data_folder)
    else:
        download_from_dataset(dataset_name, output_folder)

    meta_file = '_meta.yaml'
    write_meta(dataset_name, os.path.join(output_folder, meta_file))
    print(f"Meta data file '{meta_file}'")


if __name__ == '__main__':
    fire.Fire(download_dataset)
