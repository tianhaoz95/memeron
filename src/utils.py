import os
import uuid
import json
import numpy as np
import urllib.request
from .config import *
from scipy.ndimage import imread
from scipy.misc import imresize
from random import shuffle
from socket import timeout
from random import randint
import uuid
import validators

def download_img(url, path):
    try:
        print("downloading from " + url + " ...")
        f = urllib.request.urlopen(url=url, timeout=5)
        base, extension = os.path.splitext(url)
        with open(path + '.jpg', "wb") as img_file:
            img_file.write(f.read())
        print("image downloaded")
    except:
        print("cannot download this image")
        print("removing this sample from meta list ...")
        remove_metadata(url)

def update_metadata():
    metadata_list = json.load(open(meta_dir))
    for i in range(len(metadata_list['train'])):
        if 'uid' not in metadata_list['train'][i]:
            metadata_list['train'][i]['uid'] = str(uuid.uuid4())
    for i in range(len(metadata_list['test'])):
        if 'uid' not in metadata_list['test'][i]:
            metadata_list['test'][i]['uid'] = str(uuid.uuid4())
    with open(backup_dir, 'w') as f:
        json.dump(metadata_list, f, ensure_ascii=False)
    os.remove(meta_dir)
    with open(meta_dir, 'w') as f:
        json.dump(metadata_list, f, ensure_ascii=False)

def remove_metadata(url):
    metadata_list = json.load(open(meta_dir))
    for i in range(len(metadata_list['train'])):
        if url == metadata_list['train'][i]['url']:
            del metadata_list['train'][i]
            break
    for i in range(len(metadata_list['test'])):
        if url == metadata_list['test'][i]['url']:
            del metadata_list['test'][i]
            break
    with open(backup_dir, 'w') as f:
        json.dump(metadata_list, f, ensure_ascii=False)
    os.remove(meta_dir)
    with open(meta_dir, 'w') as f:
        json.dump(metadata_list, f, ensure_ascii=False)

def add_metadata(url_list, sample_type):
    metadata_list = json.load(open(meta_dir))
    for url in url_list:
        url = url.strip()
        if validators.url(url):
            print('adding ', url, ' ...')
            sample = {'url': url, 'type': sample_type, 'uid': str(uuid.uuid4())}
            rint = randint(0, 9)
            if rint > 3:
                metadata_list['train'].append(sample)
            else:
                metadata_list['test'].append(sample)
    with open(backup_dir, 'w') as f:
        json.dump(metadata_list, f, ensure_ascii=False)
    os.remove(meta_dir)
    with open(meta_dir, 'w') as f:
        json.dump(metadata_list, f, ensure_ascii=False)

def update_dataset():
    print('updating dataset ...')
    if not os.path.exists(dataset_dir):
        print('making dataset folder ...')
        os.makedirs(dataset_dir)
    img_list = os.listdir(dataset_dir)
    file_list = []
    print('scanning dataset ...')
    for img_file in img_list:
        filename, extension = os.path.splitext(img_file)
        file_list.append(filename)
    file_list = set(file_list)
    metadata = json.load(open(meta_dir))
    meta_list = []
    meta_list.extend(metadata['train'])
    meta_list.extend(metadata['test'])
    for meta in meta_list:
        if meta['uid'] not in file_list:
            download_img(meta['url'], dataset_dir + '/' + meta['uid'])
    print('dataset updated')

def load_img(filename, resolution):
    path = dataset_dir + '/' + filename
    print('reading image from ', path)
    img = imread(fname=path)
    print('original resolution: ', img.shape)
    print('resizing the image ...')
    resized = imresize(img, resolution)
    print('image resized to ', resized.shape)
    return resized

def load_sample(meta):
    filename = meta['uid'] + '.jpg'
    label = label_encoder[meta['type']]
    img = load_img(filename, resolution)
    sample = {'img': img, 'label': label}
    return sample

def load_data():
    img_list = os.listdir(dataset_dir)
    metadata = json.load(open(meta_dir))
    train_metadata = metadata['train']
    test_metadata = metadata['test']
    train_set = []
    test_set = []
    for train_meta in train_metadata:
        try:
            sample = load_sample(train_meta)
            train_set.append(sample)
        except:
            print('cannot load this training sample')
    for test_meta in test_metadata:
        try:
            sample = load_sample(test_meta)
            test_set.append(sample)
        except:
            print('cannot load this testing sample')
    print('loaded ', len(train_set), ' training samples')
    print('loaded ', len(test_set), ' testing samples')
    return train_set, test_set

def load_dataset():
    train_set, test_set = load_data()
    train_x, train_y = preprocess_data(train_set)
    test_x, test_y = preprocess_data(test_set)
    print('training feature shape: ', train_x.shape)
    print('training label shape: ', train_y.shape)
    print('testing feature shape: ', test_x.shape)
    print('testing label shape: ', test_y.shape)
    return train_x, train_y, test_x, test_y

def preprocess_data(data):
    sample_list = data[:]
    shuffle(sample_list)
    set_x = []
    set_y = []
    for sample in sample_list:
        x = sample['img']
        y = sample['label']
        if x.shape == conv_input_shape:
            set_x.append(x)
            set_y.append(y)
    output_x = np.array(set_x)
    output_y = np.array(set_y)
    return output_x, output_y

def random_sample(x, y, sample_size):
    size = x.shape[0]
    sampled_x = []
    sampled_y = []
    for i in range(sample_size):
        idx = randint(0, size-1)
        sampled_x.append(x[idx,:])
        sampled_y.append(y[idx,:])
    output_x = np.array(sampled_x)
    output_y = np.array(sampled_y)
    return output_x, output_y
