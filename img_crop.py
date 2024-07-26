import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['Birds', 'StanfordCars', 'StanfordDogs'])
parser.add_argument('--root', default='/data/toys')
args = parser.parse_args()

from skimage.io import imread, imsave
from skimage.color import gray2rgb
from scipy.io import loadmat
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os

if args.dataset == 'Birds':
    root = os.path.join(args.root, 'CUB_200_2011')
    images = [line.split()[1] for line in open(os.path.join(root, 'images.txt')) if len(line) > 1]
    bboxes = [[int(float(v)) for v in line.split()[1:]] for line in open(os.path.join(root, 'bounding_boxes.txt'))]
    splits = {}
    for split, value in [('train', 1), ('test', 0)]:
        ix = [int(line.split()[0])-1 for line in open(os.path.join(root, 'train_test_split.txt')) if len(line) > 1 and int(line.split()[1]) == value]
        splits[split] = []
        for i in ix:
            klass = os.path.dirname(images[i])
            dname = os.path.join(os.path.join(root, 'images', os.path.dirname(images[i])))
            fname = os.path.basename(images[i])
            bbox = (bboxes[i][0], bboxes[i][1], bboxes[i][0]+bboxes[i][2], bboxes[i][1]+bboxes[i][3])
            splits[split].append((klass, dname, fname, bbox))
if args.dataset == 'StanfordCars':
    root = os.path.join(args.root, 'stanford_cars')
    class_names = list(loadmat(os.path.join(root, 'devkit', 'cars_meta.mat'), simplify_cells=True)['class_names'])
    splits = {}
    for split in ['train', 'test']:
        fname = 'devkit/cars_train_annos.mat' if split == 'train' else 'cars_test_annos_withlabels.mat'
        data = loadmat(os.path.join(root, fname), simplify_cells=True)['annotations']
        splits[split] = []
        for d in data:
            klass = class_names[d['class']-1]
            dname = os.path.join(root, f'cars_{split}')
            fname = d['fname']
            bbox = (d['bbox_x1'], d['bbox_y1'], d['bbox_x2'], d['bbox_y2'])
            splits[split].append((klass, dname, fname, bbox))
if args.dataset == 'StanfordDogs':
    root = os.path.join(args.root, 'stanford_dogs')
    class_names = ['-'.join(species.split('-')[1:]) for species in sorted(os.listdir(os.path.join(root, 'Images')))]
    splits = {}
    for split in ['train', 'test']:
        data = loadmat(os.path.join(root, f'{split}_list.mat'), simplify_cells=True)
        splits[split] = []
        for fname, klass in zip(data['file_list'], data['labels']-1):
            klass = class_names[klass]
            folder = os.path.dirname(fname)
            dname = os.path.join(root, 'Images', folder)
            fname = os.path.basename(fname)
            ann = ET.parse(os.path.join(root, 'Annotation', folder, fname[:-4]))
            bboxes = [{c.tag: int(c.text) for c in bbox} for bbox in ann.findall('.//bndbox')]
            bbox = (
                min(bb['xmin'] for bb in bboxes),
                min(bb['ymin'] for bb in bboxes),
                max(bb['xmax'] for bb in bboxes),
                max(bb['ymax'] for bb in bboxes),
            )
            splits[split].append((klass, dname, fname, bbox))

for name, split in splits.items():
    full_path = os.path.join('datasets', args.dataset, name)
    crop_path = os.path.join('datasets', args.dataset, name + '_cropped')
    os.makedirs(full_path)
    os.makedirs(crop_path)
    for klass, dname, fname, bbox in tqdm(split):
        image = imread(os.path.join(dname, fname))
        if len(image.shape) == 2:
            image = gray2rgb(image)
        image = image[..., :3]  # RGBA to RGB
        crop_image = image[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
        os.makedirs(os.path.join(full_path, klass), exist_ok=True)
        imsave(os.path.join(full_path, klass, fname), image)
        os.makedirs(os.path.join(crop_path, klass), exist_ok=True)
        imsave(os.path.join(crop_path, klass, fname), crop_image)
