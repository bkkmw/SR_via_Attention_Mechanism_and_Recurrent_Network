import torchvision
import torchvision.transforms as transforms
import os

# My dataset loading function
def make_dataset(root, test) -> list:
    dataset = []
    # sub folder names of data set
    if test is True:
        src_dir = 'test_A'
        matt_dir = 'test_B'
        free_dir = 'test_C'
    else:
        src_dir = 'train_A'
        matt_dir = 'train_B'
        free_dir = 'train_C'

    # file names of dataset
    src_fnames = sorted(os.listdir(os.path.join(root, src_dir)))
    matt_fnames = sorted(os.listdir(os.path.join(root, matt_dir)))
    free_fnames = sorted(os.listdir(os.path.join(root, free_dir)))

    # matching datasets by name
    # same fname for triplets
    for src_fname in src_fnames:
        # source image (image with shadow)
        src_path = os.path.join(root, src_dir, src_fname)
        if src_fname in matt_fnames:
            # shadow matte image
            matt_path = os.path.join(root, matt_dir, src_fname)
            if src_fname in free_fnames:
                # shadow free image
                free_path = os.path.join(root, free_dir, src_fname)
                # if triplets exists append to dataset
                temp = (src_path, matt_path, free_path)
                dataset.append(temp)
            # if one of triplets missing do NOT append to dataset
            else:
                print(src_fname, 'Shadow free file missing')
                continue
        else:
            print(src_fname, 'Shadow matte file missing')
            continue

    return dataset


class ARGAN_Dataset(torchvision.datasets.vision.VisionDataset):
    # ARGAN dataset class composed of 3 func
    def __init__(self, root, loader=torchvision.datasets.folder.default_loader,
                 is_test=False, src_trans=None, matt_trans=None):
        super().__init__(root, transform=src_trans, target_transform=matt_trans)

        # Custom dataset loader for Training
        samples = make_dataset(self.root, test=is_test)
        self.loader = loader
        self.samples = samples
        # train data list

    #    self.src_samples = [s[0] for s in samples]
    #    self.matt_samples = [s[1] for s in samples]
    #    self.free_samples = [s[2] for s in samples]

    # Get single data
    def __getitem__(self, index):
        # load training data
        src_path, matt_path, free_path = self.samples[index]
        src_sample = self.loader(src_path)
        matt_sample = self.loader(matt_path)
        free_sample = self.loader(free_path)

        # transform data if required
        if self.transform is not None:
            # transform for RGB image : Shadow image and Shadow free image
            src_sample = self.transform(src_sample)
            free_sample = self.transform(free_sample)
        if self.target_transform is not None:
            # transform for Binary image : Shaode Matte
            matt_sample = self.target_transform(matt_sample)

        return src_sample, matt_sample, free_sample

    # Get dataset length
    def __len__(self):
        return len(self.samples)


# image Transforms
# image size 256x256 used for training _ from paper
img2tensor = transforms.Compose([
                                 transforms.Resize(size=(256,256)),
                                 transforms.ToTensor()
                                 # additional tasks
])
matt2tensor = transforms.Compose([
                                  transforms.Resize(size=(256,256)),
                                  transforms.Grayscale(1),
                                  transforms.ToTensor()
                                  # additional tasks

])