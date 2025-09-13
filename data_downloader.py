import flickrapi
import requests
import os
import json
import argparse

def get_photo_id_from_filename(filename):
    """
    Extract the photo ID from the Flickr filename.
    The photo ID is the numeric part before the underscore in the filename.
    Example: '4603553409_5b5f2b63b3.jpg' => '4603553409'
    """
    return filename.split('_')[0]

def download_image(flickr, photo_id, filename, output_dir):
    """
    Download an image from Flickr given its photo ID and save it locally with the original filename.
    """
    try:
        # Get sizes available for the photo
        sizes = flickr.photos.getSizes(photo_id=photo_id)
        original_size = sizes['sizes']['size'][-1]['source']  # Get the original size URL

        # Download the image
        response = requests.get(original_size)
        if response.status_code == 200:
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'wb') as img_file:
                img_file.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def main(api_key, api_secret, metadata_path, output_dir):
    # Create a Flickr API instance
    flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the metadata with filenames
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Iterate through the metadata and download images
    for item in metadata:
        filename = item['ground_path']
        photo_id = get_photo_id_from_filename(filename)
        download_image(flickr, photo_id, filename, output_dir)
    
    print("Download complete.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download Flickr images using filenames.")
    parser.add_argument('--api_key', type=str, required=True, help='Flickr API key.')
    parser.add_argument('--api_secret', type=str, required=True, help='Flickr API secret.')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to the metadata JSON file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where images will be downloaded.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.api_key, args.api_secret, args.metadata_path, args.output_dir)
