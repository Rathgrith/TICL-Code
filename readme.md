## Time-Image Contrastive Learning

### About

This is a repo containing code and dataset utilities for the TMLR paper [What Time Tells Us? An Explorative Study of Time Awareness Learned from Static Images](https://openreview.net/pdf?id=f1MYOG4iDG).

![](https://github.com/Rathgrith/TICL-Code/blob/main/assets/image.png?raw=true)

### Usage

#### Preparing TOC Dataset

The metadata for TOC Dataset is already under ```./metadata```, which are ```train_metadata.json``` and ```test_metadata.json``` accordingly.

We provided a script to download images from Flickr, which requires [Flickr api](https://www.flickr.com/services/api/) key. To run it, make sure you have the required metadata for test set and training set, simply call for train set and test set:

```python
python data_downloader.py --api_key your_api_key --api_secret your_api_secret --metadata_path your_metadata_path --output_dir "your_image_folder"
```

We also released the whole dataset including TOC dataset and additional AMOS-test dataset after the review process. Please download them here: [TOC](https://kaggle.com/datasets/011af1d77cea3112779e0ea0139debab55141b1dd93d0c2524cfc68ec5be774d), [AMOS](https://www.kaggle.com/datasets/rathgrith/amos-time-estimation-test/data).

#### Precompute CLIP features(Recommended)

We provided a script to precompute CLIP features. Using precomputed features can significantly reduce training time. To do it, simply call:

```python
python precompute_features.py --batch_size 32 --metadata "/path/to/metadata.json" --dir "/path/to/images" --output "/path/to/precomputed_features.h5"
```

#### Pretext Tasks

Simply reuse ```demo.ipynb``` to train and test the model.

#### Downstream Tasks

Time-based retrieval is appended to the end of ```demo.ipynb``` while other downstream tasks are stored in the folder ```./other_downstreams```.
 - Retrieval: You need the query/database split given by files: 
 - Video Scene Classification: You need to download the datasets: [YUP++](https://openaccess.thecvf.com/content_ECCV_2018/papers/Isma_Hadji_A_New_Large_ECCV_2018_paper.pdf), [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and [360+x](https://x360dataset.github.io/) and prepare index json files for each of them, basically, you need two fields for each entry. ```'path'``` for video path and ```'label'``` for video classes. See details in the code.

 - Time-based Image Editing: You need to prepare pre-trained [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) models following unofficial implementations by Rosinality (Due to StyleCLIP's code dependency on that). See more details in the code.
