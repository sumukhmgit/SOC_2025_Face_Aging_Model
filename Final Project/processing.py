import os
import shutil
from argparse import ArgumentParser
from PIL import Image


def parse_args():
    parser = ArgumentParser(description="Preprocess multiple UTKFace folders into two age domains for CycleGAN training")
    parser.add_argument('--data_dir', type=str, default='data/archive',
                        help='Root directory containing UTKFace image folders')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory where processed images will be saved')
    parser.add_argument('--min_age_A', type=int, default=18, help='Minimum age for domain A (young)')
    parser.add_argument('--max_age_A', type=int, default=28, help='Maximum age for domain A (young)')
    parser.add_argument('--min_age_B', type=int, default=40, help='Minimum age for domain B (older)')
    parser.add_argument('--max_age_B', type=int, default=120, help='Maximum age for domain B (older)')
    parser.add_argument('--img_size', type=int, default=256, help='Resize all images to this size (square)')
    return parser.parse_args()


def collect_image_files(root_dir):
    """
    Gather images from all known archive subfolders.
    Supports: crop_part1, UTKFace, utkface_aligned_cropped/crop_part1, utkface_aligned_cropped/UTKFace
    """
    valid_exts = ('.jpg', '.jpeg', '.png')
    folders = [
        'crop_part1',
        'UTKFace',
        os.path.join('utkface_aligned_cropped', 'crop_part1'),
        os.path.join('utkface_aligned_cropped', 'UTKFace')
    ]
    entries = []
    for sub in folders:
        path = os.path.join(root_dir, sub)
        if not os.path.isdir(path):
            continue
        for fname in os.listdir(path):
            if fname.lower().endswith(valid_exts):
                entries.append((path, fname))
    return entries


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # collect all image entries
    images = collect_image_files(args.data_dir)
    print(f"Found {len(images)} images under {args.data_dir}")

    # parse ages
    parsed = []
    for folder, fname in images:
        try:
            age = int(fname.split('_')[0])
            parsed.append((folder, fname, age))
        except ValueError:
            continue

    # split domains
    domainA = [(fpath, name) for fpath, name, age in parsed if args.min_age_A <= age <= args.max_age_A]
    domainB = [(fpath, name) for fpath, name, age in parsed if args.min_age_B <= age <= args.max_age_B]

    # balance count
    N = min(len(domainA), len(domainB))
    domainA = sorted(domainA, key=lambda x: x[1])[:N]
    domainB = sorted(domainB, key=lambda x: x[1])[:N]
    print(f"Domain A: {len(domainA)} images  |  Domain B: {len(domainB)} images")

    # output dirs
    dirA = os.path.join(args.output_dir, 'trainA')
    dirB = os.path.join(args.output_dir, 'trainB')
    os.makedirs(dirA, exist_ok=True)
    os.makedirs(dirB, exist_ok=True)

    # copy & resize
    for folder, fname in domainA:
        src = os.path.join(folder, fname)
        dst = os.path.join(dirA, fname)
        with Image.open(src) as img:
            img = img.convert('RGB')
            img = img.resize((args.img_size, args.img_size), Image.LANCZOS)
            img.save(dst)
    for folder, fname in domainB:
        src = os.path.join(folder, fname)
        dst = os.path.join(dirB, fname)
        with Image.open(src) as img:
            img = img.convert('RGB')
            img = img.resize((args.img_size, args.img_size), Image.LANCZOS)
            img.save(dst)

    print(f"Saved processed images to {args.output_dir}/trainA and trainB")

if __name__ == '__main__':
    main()
