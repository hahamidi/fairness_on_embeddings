
import torch
from torch.utils.data import Dataset
import numpy as np
from imageio.v2 import imread
from PIL import Image
import pandas as pd


class ImageXrayDataset(Dataset):
    def __init__(self, dataframe_path, path_image, finding="any", transform=None):
        self.dataframe = pd.read_csv(dataframe_path)
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.finding = finding
        self.transform = transform
        # "/datasets/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
        self.path_image = path_image

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.dataframe.columns:
                if len(self.dataframe[self.dataframe[finding] == 1]) > 0:
                    self.dataframe = self.dataframe[self.dataframe[finding] == 1]
                else:
                    print("No positive cases exist for " + finding + ", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")
        self.PRED_LABEL = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Lesion',
            'Lung Opacity',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

    def __getitem__(self, idx):
    


        item = self.dataframe.iloc[idx]
        
        
        img = imread(self.path_image + item["path"] + '.jpg')
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        for i in range(0, len(self.PRED_LABEL)):

            if (self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')

        sample = {'data':img, 'labels': np.array(list(label))}

        return sample


    def __len__(self):
        return self.dataset_size

class ImageXraySensitive(Dataset):
    def __init__(self, dataframe_path, path_image, finding="any", transform=None , sensitive_label = 'gender', sensitive_values = ['M', 'F']):
        
        self.sensitive_label = sensitive_label
        self.sensitive_values = sensitive_values
        self.dataframe = pd.read_csv(dataframe_path)
        # drop rows with values not in sensitive_values
        self.dataframe = self.dataframe[self.dataframe[self.sensitive_label].isin(sensitive_values)]
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.finding = finding
        self.transform = transform
        # "/datasets/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
        self.path_image = path_image


    def __getitem__(self, idx):
    


        item = self.dataframe.iloc[idx]
        
        
        img = imread(self.path_image + item["path"] + '.jpg')
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        label = torch.FloatTensor(np.zeros(len(self.sensitive_values), dtype=float))
        for i in range(0, len(self.sensitive_values)):
            if (self.dataframe[self.sensitive_label].iloc[idx] == self.sensitive_values[i]):
                label[i] = 1
        
        sample = {'data':img, 'labels': np.array(list(label))}
        return sample
    
    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    # Test the class
    dataset = ImageXrayDataset(dataframe_path="./dataset/mimic/mimic_test_df.csv", path_image="/datasets/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/", finding="any")
    print(dataset[0])


