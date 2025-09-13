import argparse
import os
import torch
from PIL import Image
import numpy as np
import h5py
import json

from utils import read_data_and_metadata, compute_clip_features
from torchvision import transforms
from torch.utils.data import DataLoader

from model.TICL import ImageEncoder
from data.Dataloader import CustomDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and save CLIP features.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for dataloaders')
    parser.add_argument('--metadata', type=str, help='Metadata file for training data')
    parser.add_argument('--dir', type=str, help='Directory with training images')
    parser.add_argument('--output', type=str, default= 'clip_features_train.h5', help='Output file for train computed features')

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(metadata_file=args.metadata, root_dir=args.dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    compute_clip_features(ImageEncoder(), dataloader, args.output)

    # check the saved features
    hdf5_file_path = args.output
    index = 0 
    print(f"Checking the saved features at index {index}")
    features, metadata = read_data_and_metadata(hdf5_file_path, index)
    timestamp = metadata.get("time")

    print("Features shape:", features.shape)
    print("Metadata:", metadata)
    print("Timestamp:", timestamp)
    print("Timezone:", metadata.get("timezone"))
    # check sample counts in the h5 file and the metadata file
    with h5py.File(hdf5_file_path, 'r') as h5f:
        num_samples = len(h5f['features'])
        print(f"Number of samples in the h5 file: {num_samples}")
        print(f"Number of samples in the metadata file: {len(json.load(open(args.metadata)))})")
        if num_samples != len(json.load(open(args.metadata))):
            print("Number of samples in the h5 file and the metadata file do not match.")
        else:
            print("Number of samples in the h5 file and the metadata file match.")