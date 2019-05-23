import os
import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

from os import listdir
from os.path import isfile, join


#returns all ids of the images in that directory
def give_ids(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    result = []

    for file in onlyfiles:
        if(('.xml' in file) or ('.jpg' in file)):
            if (file[:-4] not in result):
                result.append(file[:-4])
    return result

#return an array of 4 values, xmax, xmin, ymax,ymin for a path
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = []
    labels = list()
    for object in root.iter('object'):


        label = object.find('name').text.lower().strip()

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append(xmin)
        boxes.append(ymin)
        boxes.append(xmax)
        boxes.append(ymax)

    return boxes

#saves a text file of all the ids from all the dir
def generate_trailval(path):
    cur_path = os.getcwd()
    path = 'D:/training data'
    os.chdir(path)
    items = os.listdir()
    print("doing ...")
    os.chdir(cur_path)
    results = []
    for x in range (0,len(items)):
        items[x] = "D:/training data/" + items[x]
    for item in items:
        results.append(give_ids(item))

    file = open('trainval.txt', 'w')

    for result in results:
        print(result)
        for item in result: 
            file.write(item + '\n')

    file.close()

    print("done!")


def create_data_lists(src_path, out_path):
    path = os.path.abspath(src_path)
    train_images = []
    train_object = []

    #gather all the ids in the trainval.txt
    with open(os.path.join(path, 'trainval.txt')) as f:
        ids = f.read().splitlines()
    print('gathering ids')
    for id in ids:
        print(id)
        #read it from xml file
        objects = parse_annotation(os.path.join(src_path, 'labels/', id + '.xml'))

        train_object.append(objects)
        train_images.append(os.path.join(src_path,'images',id + '.jpg'))

    #make sure they have the same number for both images and gt
    assert(len(train_object) == len(train_images))
    print('saving...')
    #save to file
    with open(os.path.join(out_path,'train_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(out_path, 'train_objects.json'), 'w') as j:
        json.dump(train_object, j)

    #summary
    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images),len(train_object), os.path.abspath(out_path)))


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def transform(image, boxes, split):
    assert split in {'TRAIN', 'TEST'}

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations if validation/evaluation
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        #new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        #if random.random() < 0.5:
        #    new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        #new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
        #                                                                 new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        #if random.random() < 0.5:
        #    new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    #new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    return new_image, new_boxes
