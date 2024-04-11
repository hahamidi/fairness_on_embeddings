from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import numpy as np


class VecDataset_All(Dataset):
    def __init__(self, info_df_path_mimic, info_df_path_chexpert, data_path_mimic, data_path_chexpert):
        self.PRED_LABEL= [
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


        self.info_df_mimic = pd.read_csv(info_df_path_mimic)
        self.info_df_chexpert = pd.read_csv(info_df_path_chexpert)
        self.lables_mimic = self.info_df_mimic[self.PRED_LABEL]
        self.lables_chexpert = self.info_df_chexpert[self.PRED_LABEL]
        self.samples_path_mimic = self.info_df_mimic['path']
        self.samples_path_chexpert = self.info_df_chexpert['path']
        self.data_path_mimic = data_path_mimic
        self.data_path_chexpert = data_path_chexpert
        self.len_mimic = len(self.info_df_mimic)
        self.len_chexpert = len(self.info_df_chexpert)

    def __len__(self):

        return self.len_mimic + self.len_chexpert
    
    def __getitem__(self, idx):

        if idx < self.len_mimic:
            
            path = self.samples_path_mimic.iloc[idx]
            path = path + ".npy"
            full_item_path = self.data_path_mimic + path
            item = np.load(full_item_path)
            #convert to tensor
            item = torch.from_numpy(item)
            # read labels from labels_df
            label = self.lables_mimic.iloc[idx]
            label = list(label)
            label = torch.tensor(label, dtype=torch.float32)

       
            return {'data':item, 'labels': label}
        else:
            
            idx = idx - len(self.info_df_mimic)
            path = self.samples_path_chexpert.iloc[idx]
            path = path.replace("/", "_").replace(".", "_") + ".npy"
            # first char of path to / instead of _
            path = "/" + path[1:]
            full_item_path = self.data_path_chexpert + path
            item = np.load(full_item_path)
            #convert to tensor
            item = torch.from_numpy(item)
            # read labels from labels_df
            label = self.lables_chexpert.iloc[idx]
            label = list(label)
            label = torch.tensor(label, dtype=torch.float32)
    
            return {'data':item, 'labels': label}


class VectorXrayDataset(Dataset):
    def __init__(self, dataframe_path, path_vector, finding="any"):
        self.dataframe = pd.read_csv(dataframe_path)
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.finding = finding
        # "/datasets/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
        self.path_vector = path_vector
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
            vector = np.load(self.path_vector + item["path"] + '.npy')
            label = np.zeros(len(self.PRED_LABEL), dtype=int)
            label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
            for i in range(0, len(self.PRED_LABEL)):
                if (self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                    label[i] = self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')
            sample = {'data':vector, 'labels': np.array(list(label))}

            return sample


    def __len__(self):
            return self.dataset_size
class VectorXraySensitiveDataset(Dataset):
    def __init__(self, dataframe_path, path_vector, finding="any", sensitive_label="gender", sensitive_values=['M', 'F']):
        self.dataframe = pd.read_csv(dataframe_path)
        # drop rows with values not in sensitive_values
        self.dataframe = self.dataframe[self.dataframe[sensitive_label].isin(sensitive_values)]
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.finding = finding
        self.sensitive_label = sensitive_label
        self.sensitive_values = sensitive_values
        # "/datasets/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
        self.path_vector = path_vector

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        vector = np.load(self.path_vector + item["path"] + '.npy')

        label = torch.FloatTensor(np.zeros(len(self.sensitive_values), dtype=float))
        for i in range(0, len(self.sensitive_values)):
            if (self.dataframe[self.sensitive_label].iloc[idx] == self.sensitive_values[i]):
                label[i] = 1
        sample = {'data':vector, 'labels': np.array(list(label))}
        return sample
    
    def __len__(self):
        return self.dataset_size
        


if __name__ == "__main__":
    # Test the class
    dataset = VectorXrayDataset(dataframe_path="./dataset/mimic/mimic_test_df.csv", path_vector="/fs01/home/hhamidi/diff/fariness_embedding_second_submit/fairness_on_embeddings/dataset/vector_embeddings/", finding="any")
    print(dataset[0])
    dataset2 = VectorXrayDataset(dataframe_path="./dataset/chexpert/cxp_train_df.csv", path_vector="/fs01/home/hhamidi/diff/fariness_embedding_second_submit/fairness_on_embeddings/dataset/vector_embeddings/", finding="any")
    print(dataset2[10])
    dataset3 = VectorXraySensitiveDataset(dataframe_path="./dataset/mimic/mimic_test_df.csv", path_vector="/fs01/home/hhamidi/diff/fariness_embedding_second_submit/fairness_on_embeddings/dataset/vector_embeddings/", finding="any",
    sensitive_label="gender", sensitive_values=['M', 'F'])
    print(dataset3[1])