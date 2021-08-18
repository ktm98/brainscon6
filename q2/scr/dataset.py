from torch.utils.data import Dataset
import cv2
import numpy as np


def get_image(file_path):
    image = cv2.imread(file_path)[:, :, ::-1]
    # image = image.astype(np.float32)
    # image = np.vstack(image).transpose((1, 0))
    return image

class ImageDataset(Dataset):
    def __init__(self, file_names, labels=None, transform=None, return_dict=False):
        """
        Args:
            file_names
            labels
            transform
            return_dict  (bool): if True, __getitem__ returns dict type, else, returns tuple
        """
        self.file_names = file_names
        self.labels = labels
        self.transform = transform
        self.return_dict = return_dict
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = get_image(file_path)
        if self.transform:
            image = self.transform(image=image)['image']
        if self.labels is None:
            if self.return_dict:
                return {'image': image}
            else:
                return image
            
        else:
            label = self.labels[idx]
            if self.return_dict:
                return {'image': image, 'label': label}
            else:
                return image, label

    
class SegmentationDataset(Dataset):

    CLASSES = ['background', 'p_con']

    def __init__(self, image_file_names, mask_file_names=None, labels=None, classes=None, transform=None, return_dict=False, preprocess=None):
        """
        Args:
            file_names
            labels
            transform
            return_dict  (bool): if True, __getitem__ returns dict type, else, returns tuple
        """
        self.image_file_names = image_file_names
        self.mask_file_names = mask_file_names
        self.labels = labels
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]


        self.transform = transform
        self.preprocess = preprocess
        self.return_dict = return_dict
        
    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        file_path = self.image_file_names[idx]

        image = get_image(file_path)

        if self.mask_file_names is not None:
            mask_file_path = self.mask_file_names[idx]
            mask = cv2.imread(mask_file_path, 0)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            if self.preprocess:
                preprocessed = self.preprocess(image=image, mask=mask)
                image = preprocessed['image']
                mask = preprocessed['mask']
            if self.labels is not None:
                labels = self.labels[idx]
            if self.return_dict:
                if self.labels is not None:
                    return {'image': image, 'mask': mask, 'label': labels}
                else:
                    return {'image': image, 'mask': mask}
            else:
                if self.labels is not None:
                    return image, mask, labels
                else:
                    return image, mask
        else:
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            if self.preprocess:
                preprocessed = self.preprocess(image=image)
                image = preprocessed['image']

            if self.labels is not None:
                labels = self.labels[idx]
            if self.return_dict:
                if self.labels is not None:
                    return {'image': image, 'label': labels}
                else:
                    return {'image': image}
            else:
                if self.labels is not None:
                    return image, labels
                else:
                    return image       



class ConcatImageDataset(Dataset):
    CLASSES = ['background', 'p_con']

    def __init__(self, image_file_names, mask_file_names=None, classes=None, transform=None, return_dict=False, preprocess=None):
        """
        Args:
            file_names
            labels
            transform
            return_dict  (bool): if True, __getitem__ returns dict type, else, returns tuple
        """
        self.image_file_names = image_file_names
        self.mask_file_names = mask_file_names
        # self.labels = labels
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]


        self.transform = transform
        self.preprocess = preprocess
        self.return_dict = return_dict
        
    def __len__(self):
        return len(self.image_file_names)


    def __getitem__(self, idx):
        file_path = self.image_file_names[idx]

        image = get_image(file_path)

        if self.mask_file_names is not None:
            mask_file_path = self.mask_file_names[idx]
            mask = cv2.imread(mask_file_path, 0)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            labels = (mask.sum() > 0).astype(np.float32)
            if self.preprocess:
                preprocessed = self.preprocess(image=image, mask=mask)
                image = preprocessed['image']
                mask = preprocessed['mask']
            # if self.labels is not None:
            
            if self.return_dict:
                return {'image': image, 'mask': mask, 'label': labels}
            else:
                return image, mask, labels
        else:
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            if self.preprocess:
                preprocessed = self.preprocess(image=image)
                image = preprocessed['image']


            if self.return_dict:

                return {'image': image}
            else:

                return image 