import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from imageio import imread
from PIL import Image
import torchvision.transforms as tfs
import cv2





class ImageDataset_all(Dataset):
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
            'No_Finding',
            'Enlarged_Cardiomediastinum',
            'Cardiomegaly',
            'Lung_Lesion',
            'Lung_Opacity',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural_Effusion',
            'Pleural_Other',
            'Fracture',
            'Support_Devices']

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
    

    

class ImageDatasetOneFolder(Dataset):
    def __init__(self, data_folder_path,discovery ='/*' ,split_name='all' , split_ratio = 0.8, transform=None):
        self.all_path = glob.glob(data_folder_path + discovery)
        self.all_path.sort()
        if split_name == 'train':
            self.all_path = self.all_path[:int(len(self.all_path) * split_ratio)]
        elif split_name == 'val':
            self.all_path = self.all_path[int(len(self.all_path) * split_ratio):]
        
        self.transform = transform


        # extract labels from the path
        self.numerical_labels =  np.zeros((len(self.all_path), 14))
        for i in range(len(self.all_path)):


            label = [float(l) for l in self.all_path[i].split('_')[-1].replace('.jpg', '').replace('.png', '').replace('[', '').replace(']', '').split(',')]


            if len(label) == 13:
                # if all labels are zero, then it is a no finding and add the last label to be 1
                if sum(label) == 0:
                    label.append(1)
                else:
                    label.append(0)
            
                
            self.numerical_labels[i] = np.array(label)
        
        if split_name == 'val':
            # shuffle labels
            shuffled_indices = np.random.permutation(len(self.numerical_labels))
            self.numerical_labels = self.numerical_labels[shuffled_indices]



     

        self.PRED_LABEL = [
            'Enlarged_Cardiomediastinum',
            'Cardiomegaly',
            'Lung_Lesion',
            'Lung_Opacity',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural_Effusion',
            'Pleural_Other',
            'Fracture',
            'Support_Devices']
        # sort of the PRED_LABEL
        self.PRED_LABEL.sort()
        self.PRED_LABEL.append('No_Finding')
        


    def __getitem__(self, idx):

        img = imread(self.all_path[idx])
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.numerical_labels[idx]
        # label = np.array(label[-1][np.newaxis])
        # add a dimension to the label
    

        sample = {'data':img, 'labels':label }

        return sample


    def __len__(self):
        return len(self.all_path)


class Chexpert(Dataset):
    def __init__(self, dataframe_path, path_image, transform=None , MAPPING = {-1: -1, 0: 0, 1: 1, np.nan: 0}):
        self.dataframe = pd.read_csv(dataframe_path)
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        # "/datasets/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
        self.path_image = path_image
        self.LABEL_NAMES =['Atelectasis',
                            'Cardiomegaly',
                            'Consolidation',
                            'Edema',
                            'Enlarged Cardiomediastinum',
                            'Fracture',
                            'Lung Lesion',
                            'Lung Opacity',
                            'No Finding',
                            'Pleural Effusion',
                            'Pleural Other',
                            'Pneumonia',
                            'Pneumothorax',
                            'Support Devices']
        
        self.MAPPING = MAPPING
        # 1 for the presence of the condition,
        # 0 for the absence,
        # -1 for uncertainty,
        # nan if the condition was not evaluated or the label is missing.

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        img_path = os.path.join(self.path_image, item["Path"])
        img = imread(img_path)

        if len(img.shape) == 2:  # Handle grayscale images
            img = np.stack([img] * 3, axis=-1)
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.FloatTensor(item[self.LABEL_NAMES].replace(self.MAPPING).values)

        sample = {
            'data': img,
            'labels': label,
            'non_mapped_labels': np.array(item[self.LABEL_NAMES].values, dtype=float)
        }

        return sample


class CheXpert_Advanced(Dataset):
    '''
    Reference: 
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''
    def __init__(self, 
                 csv_path, 
                 image_root_path='',
                 image_size=320,
                 class_index=0, 
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion'],
                 mode='train',
                 split_size=1.0):
        
    
        # load data from csv
        print(image_root_path)
        print ('Loading data from %s...'%csv_path)
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal and 'Frontal/Lateral' in self.df.columns:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']  
        if 'train/val/test' in self.df.columns:
            self.df = self.df[self.df['train/val/test'] == mode]
       
        if split_size < 1.0:
            self.df = self.df.sample(frac=split_size, random_state=seed)
            print(f'data set splited with size of {split_size}')
        
        
            
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print ('Upsampling %s...'%col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)


        # impute missing values 
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)  
                self.df[col].fillna(0, inplace=True) 
            elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)
        
        
        self._num_images = len(self.df)
        
        # 0 --> -1
        if flip_label and class_index != -1: # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)   
            
        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
        
        
        
        assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1: # 5 classes
            print ('Multi-label mode: True, Number of classes: [%d]'%len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):

                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
  



                self.value_counts_dict[class_key] = class_value_counts_dict
 
        else:       # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()
        self.mode = mode
        self.class_index = class_index
        self.image_size = image_size

        
        self._images_list =  [image_root_path+path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()
    
        if verbose:
            if class_index != -1:
                print ('-'*30)
                if flip_label:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[-1]+self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                else:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[0]+self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[0] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                print ('-'*30)
            else:
                print ('-'*30)
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    print (self.value_counts_dict[class_key], self.value_counts_dict[class_key])
                    imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                    print ()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                print ('-'*30)
            
    @property        
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)
       
    @property  
    def data_size(self):
        return self._num_images 
    
    def image_augmentation(self, image):
        img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
        image = img_aug(image)
        return image
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        if self.mode == 'train':
            image = self.image_augmentation(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)  
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ =  np.array([[[0.229, 0.224, 0.225]  ]]) 
        image = (image-__mean__)/__std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        if self.class_index != -1: # multi-class mode
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)

        sample = {'data': image, 'labels': label}
        return sample



 









if __name__ == "__main__":
        chexpert_dataset = Chexpert('/fs01/home/hhamidi/projects/stable-diffusion/data/csv_files/train.csv', '/datasets/chexpert/', transform=None)
        print(chexpert_dataset[10])
        chexpert_advanced  = CheXpert_Advanced(csv_path='/fs01/home/hhamidi/projects/stable-diffusion/data/csv_files/train.csv', image_root_path='/datasets/chexpert/CheXpert-v1.0-small/', use_upsampling=True, use_frontal=True, image_size=224, mode='train', class_index=-1)
        print(chexpert_advanced[10])

