"""
Download and save DocVQA images from HuggingFace datasets.

This script downloads images from the DocVQA dataset and saves them locally
as sample_0.png, sample_1.png, etc. for use in experiments.

Usage:
    python save_docvqa_images.py --output_dir images --num_images 100
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm


def main(output_dir: str, num_images: int = 100, split: str = "validation"):
    """Download and save images from the DocVQA dataset.
    
    Args:
        output_dir: Directory to save images.
        num_images: Number of images to save.
        split: Dataset split to use (train, validation, test).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading DocVQA dataset (split={split})...")
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA")
    
    for i in tqdm(range(num_images), desc="Saving images"):
        sample = ds[split][i]
        image = sample["image"]
        image_path = os.path.join(output_dir, f"sample_{i}.png")
        if not os.path.exists(image_path):
            image.save(image_path)
    
    print(f"Saved {num_images} images to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DocVQA images from HuggingFace")
    parser.add_argument("--output_dir", type=str, default="images", help="Directory to save images")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to download")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    args = parser.parse_args()
    main(args.output_dir, args.num_images, args.split)
