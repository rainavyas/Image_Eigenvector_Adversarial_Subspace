import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import os
import scandir
import cv2

def process_tinyImageNetData(base_dir):
    transform_train = transforms.Compose([
          torchvision.transforms.ToPILImage(),
          torchvision.transforms.RandomCrop(size=64, padding=4),
          torchvision.transforms.RandomHorizontalFlip(p=0.25),
          torchvision.transforms.RandomRotation(15),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(
              mean=[0.4914, 0.4822, 0.4465],
              std=[0.2023, 0.1994, 0.2010],
          ),
      ])

    transform_val = transforms.Compose([
          torchvision.transforms.ToPILImage(),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(
              mean=[0.4914, 0.4822, 0.4465],
              std=[0.2023, 0.1994, 0.2010],
          ),
      ])


    train_dir = base_dir+'/train'
    val_dir = base_dir+'/val' # Dont use test dir as it is not labeled

    # Get wnids
    wnids_file_name = base_dir + '/wnids.txt'
    with open(wnids_file_name, 'r') as f:
      wnids=f.readlines()

    wnid_to_label = {}
    for i, wnid in enumerate(wnids):
      wnid_to_label[wnid.rstrip('\n')] = i

    # Create training tensors
    X_train = []
    y_train = []

    train_folder_names = [f.name for f in scandir.scandir(train_dir)]
    print(len(train_folder_names))
    for train_folder in train_folder_names:
      curr_y = wnid_to_label[train_folder]
      img_dir = train_dir +'/' + train_folder + '/images'
      img_files = [f.name for f in scandir.scandir(img_dir)]
      if len(img_files) != 500:
        print(train_folder)
      for img_file in img_files:
        image= cv2.imread(img_dir+'/'+img_file)
        image = transform_train(image)
        X_train.append(image)
        y_train.append(curr_y)

    # Convert to tensors
    y_train = torch.LongTensor(y_train)
    X_train = torch.from_numpy(np.stack(X_train))

    # Create validation dataset
    #Create image name to label id dict
    val_img_to_label = {}
    val_images_id_filename = val_dir + '/val_annotations.txt'
    with open(val_images_id_filename, 'r') as f:
      datalines=f.readlines()

    datalines = [dataline.split('\t') for dataline in datalines]
    img_names = [dataline[0] for dataline in datalines]
    wnids = [dataline[1] for dataline in datalines]

    for img_name, wnid in zip(img_names, wnids):
      label = wnid_to_label[wnid]
      val_img_to_label[img_name] = label

    X_val = []
    y_val = []

    img_dir = val_dir+'/images'
    img_files = [f.name for f in scandir.scandir(img_dir)]
    print(len(img_files))
    for img_file in img_files:
      curr_y = val_img_to_label[img_file]
      image= cv2.imread(img_dir+'/'+img_file)
      image = transform_val(image)
      X_val.append(image)
      y_val.append(curr_y)

    # Convert to tensors
    y_val = torch.LongTensor(y_val)
    X_val = torch.from_numpy(np.stack(X_val))

    chn=3
    IMG_DIM=64

    X_train = torch.reshape(X_train, (X_train.size(0), chn, IMG_DIM, IMG_DIM))
    X_val = torch.reshape(X_val, (X_val.size(0), chn, IMG_DIM, IMG_DIM))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    return train_ds, val_ds

def get_datasets(dataset, loc=None):
    '''
    Returns a torchvision dataset
    '''
    if dataset == 'cifar10':
        train_ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                torchvision.transforms.RandomCrop(size=32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]), download=True)

        eval_ds = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                torchvision.transforms.RandomCrop(size=32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]), download=True)

    elif dataset == 'cifar100':
        train_ds = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                torchvision.transforms.RandomCrop(size=32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409],
                    std=[0.2009, 0.1984, 0.2023],
                ),
            ]), download=True)

        eval_ds = datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                torchvision.transforms.RandomCrop(size=32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409],
                    std=[0.2009, 0.1984, 0.2023],
                ),
            ]), download=True)

    else:
        # Tiny ImageNet data
        train_ds, eval_ds = process_tinyImageNetData(loc)

    return train_ds, eval_ds
