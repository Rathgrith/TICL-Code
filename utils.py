from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
import json
import h5py
import torch
from PIL import Image
def read_data_and_metadata(hdf5_file_path, index):
    with h5py.File(hdf5_file_path, 'r') as h5f:
        features = h5f['features'][index]
        metadata_str = h5f['metadata'][index].decode('utf-8')
        metadata = json.loads(metadata_str) if metadata_str else {}
    return features, metadata

def compute_clip_features(model, dataloader, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    num_samples = len(dataloader.dataset)
    num_features = 768  # Assuming feature dimension from CLIP
    with h5py.File(output_file, 'w') as h5f:
        features_dset = h5f.create_dataset('features', (num_samples, num_features), dtype='f')
        metadata_dset = h5f.create_dataset('metadata', (num_samples,), dtype=h5py.special_dtype(vlen=str))
        for i, images in enumerate(tqdm(dataloader)):
            images = images[0].to(device)
            with torch.no_grad():
                features = model(images)
            for j in range(features.size(0)):
                features_dset[i * dataloader.batch_size + j] = features[j].cpu().numpy()
                metadata = dataloader.dataset.get_metadata(i * dataloader.batch_size + j)
                metadata_dset[i * dataloader.batch_size + j] = json.dumps(metadata)
    print(f"Features saved to {output_file}")