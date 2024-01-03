import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from config import Config as config

class CustomTrainDataset(Dataset):
    """
    Custom dataset for training, handling FMRI and stimulus images from selected subjects.

    Attributes:
        fmri_files (list): List of file paths for FMRI data.
        images_files (list): List of file paths for stimulus images.
        samples_per_file (int): Number of samples per file.
        total_samples (int): Total number of samples in the dataset.
        transform (callable, optional): Optional transform to be applied on an image.
    """

    def __init__(self, base_path, subjects, transform, validation):
        """
        Initialize the dataset with specified subjects.

        Args:
            base_path (str): Base path to the dataset directories.
            subjects (tuple): Tuple of subject numbers to include in the dataset (e.g., (1, 2, 5, 7)).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.fmri_files = []
        self.images_files = []
        
        # SANITY
        print('Is the Data path available : ', os.path.exists(base_path))

        for subject_num in subjects:
            # Construct folder name from subject number
            subject_folder = f'subj{int(subject_num):02d}'
            subject_path = os.path.join(base_path, subject_folder)

            if not validation:
                # FMRI Files Training files
                fmri_files = [os.path.join(subject_path, file) for file in os.listdir(subject_path) if 'nsd_train_fmriavg_nsdgeneral' in file and 'batch' in file]
                fmri_files = sorted(fmri_files, key=lambda x: int(x.split('_batch')[-1].split('.')[0]))[:-1]
                # Stimuli Image Tranining files
                images_files = [os.path.join(subject_path, file) for file in os.listdir(subject_path) if 'nsd_train_stim' in file and 'batch' in file]
                images_files = sorted(images_files, key=lambda x: int(x.split('_batch')[-1].split('.')[0]))[:-1]
            if validation:
                # FMRI Files Testing files
                fmri_files = [os.path.join(subject_path, file) for file in os.listdir(subject_path) if 'nsd_test_fmriavg_nsdgeneral' in file]
                # Stimuli Image Testing files
                images_files = [os.path.join(subject_path, file) for file in os.listdir(subject_path) if 'nsd_test_stim' in file]

            self.fmri_files.extend(fmri_files)
            self.images_files.extend(images_files)

        # Ensure all files have the same number of samples
        self.samples_per_file = len(np.load(self.fmri_files[0], mmap_mode='r'))
        self.total_samples = self.samples_per_file * len(self.fmri_files)
        self.transform = transform

    def __len__(self):
        """ Returns the total number of samples in the dataset. """
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieves an item by index.

        Args:
            idx (int): Index of the desired sample.

        Returns:
            tuple: Tuple containing the stimulus image and corresponding FMRI data.
        """
    # DATA LOADING
        # Loading Files this orders eg: 105 gives file_idx as 3 and sample_idx as 5
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file
        # Fmri and Images
        fmri = np.load(self.fmri_files[file_idx], mmap_mode='r')[sample_idx]
        image = np.load(self.images_files[file_idx], mmap_mode='r')[sample_idx]

    # Apply preprocessing for FMRI array
        # Normalize FMRI data
        if config.normalize_fmri:
            fmri_min = fmri.min()
            fmri_max = fmri.max()
            if fmri_max - fmri_min > 0:
                fmri = (fmri - fmri_min) / (fmri_max - fmri_min)
            else:
                print('Warning: Zero or constant fMRI data encountered')
        # Padding
        pad_size = config.input_dim - fmri.size
        if pad_size > 0:
            fmri = np.pad(fmri, (0, pad_size), mode='constant', constant_values=0)
        else:
            print('Models Input neurons is lower then actual fmri input size')

        # Convert to PyTorch tensors
        fmri = torch.from_numpy(fmri).float()
        image = torch.from_numpy(image).float()


        # Apply transformations for RGB Image
        if self.transform:
            image = transforms.Resize(config.output_dim)(image.permute(2,0,1))  # Ensure correct dimension order before transform

        return image, fmri
