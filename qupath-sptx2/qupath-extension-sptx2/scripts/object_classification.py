import glob
import pandas as pd
import os
import time
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import argparse
from ResNet_sz56 import resnet142
import json
import numpy as np
import torchvision.models as models
 
parser = argparse.ArgumentParser()
parser.add_argument('action', type=str, help='action')
parser.add_argument('result_file', type=str, help='result_file')
parser.add_argument('-m','--model_file', help='model file', required=False)
parser.add_argument('-i','--image_path', help='image path', required=False)
parser.add_argument("-bz", "--batch_size", type=int, default=128, help="size of the batches")
opt = parser.parse_args()
print(opt)

RETRY_TIMES = 10
RETRY_SLEEP = 1

class SingleCellImageDataset(Dataset):
    def __init__(self, image_data, 
                 image_size, 
                 image_mean, image_std,
                 # , k=5, m_list=[0,1,2,3,4]
                 ):
        'Initialization'
        
        # dataset_partitions = self.__partition(dataset, k)
        #
        # self.dataset = []
        # for m in m_list:
        #     self.dataset += dataset_partitions[m]


        self.image_data = image_data
        self.image_mean = image_mean 
        self.image_std = image_std
        
        transformObjectList = [
            T.ToTensor(),
            T.CenterCrop(image_size),
            T.Resize(image_size, antialias=True),
            T.Normalize(image_mean, image_std),     
            # T.RandomRotation(degrees=(0, 90)),
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            ]

        self.transform = T.Compose(transformObjectList)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_data)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        imageFile = self.image_data.iloc[index]['image_filepath']

        img = Image.open(imageFile)
        img = self.transform(img)
        
        return img
    
    # def __partition(self, lst, n):
    #     division = len(lst) / float(n)
    #     return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def eval(opt):
    result_json_data = {"success": False}
    
    for try_count in range(RETRY_TIMES):
        try:
            device_count = torch.cuda.device_count()
            
            free_mem_size_list = []
            
            for d in range(device_count):
                free_mem_size_list.append(torch.cuda.mem_get_info(d)[0])
                
            gpu_id = np.argmax(np.asarray(free_mem_size_list))
            device = torch.device('cuda:'+str(gpu_id))
            
            # Prepare image data
            image_filename_list = glob.glob(os.path.join(opt.image_path, '*'))
                
            image_filepath_list = []
            image_uuid_list = []
            
            for image_filename in image_filename_list:
                _, filename = os.path.split(image_filename)
                image_uuid, _ = os.path.splitext(filename)
            
                image_filepath_list.append(image_filename)
                image_uuid_list.append(image_uuid)
                
            image_df = pd.DataFrame({'image_filepath': image_filepath_list, 'Object ID':image_uuid_list})
            image_df.set_index('Object ID', inplace=True)
            
            
            model = torch.load(opt.model_file)
            param = model['parameters']
            
            # net = models.resnet18().to(device)
            # net = models.alexnet().to(device)
            # net = models.vgg16().to(device)
            # net = models.squeezenet1_0().to(device)
            # net = models.densenet161().to(device)
            # net = models.inception_v3().to(device)
            # net = models.googlenet().to(device)
            # net = models.shufflenet_v2_x1_0().to(device)
            # net = models.mobilenet_v2() .to(device)
            
            net = resnet142(param['n_classes']).to(device)
            # net = MiniVGG19_sz56(n_classes).to(device)
  
            net.load_state_dict(model['model_state'])
            net.eval()
            
            # Create datasets
            eval_dataset = SingleCellImageDataset(
                image_df, 
                image_size=param['image_size'],
                image_mean=param['image_mean'], 
                image_std=param['image_std']) 
            
            # Create dataloaders
            eval_dataloader = DataLoader(eval_dataset, opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            
            result_json_data["predicted"] = []
            with torch.no_grad():
                for images in eval_dataloader:
                    images = images.to(device)
                    outputs = net(images)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    result_json_data["predicted"].extend(predicted.to('cpu').tolist())
            
            result_json_data["success"] = True
                    
            break
        except Exception:
            print(f'Exception occurred! retry {try_count+1}/{RETRY_TIMES}')
            time.sleep(RETRY_SLEEP)

    with open(opt.result_file, "w") as fp:
        json.dump(result_json_data , fp)
        
def param(opt):
    model = torch.load(opt.model_file)
    param_json_data = model['parameters']
    
    with open(opt.result_file, "w") as fp:
        json.dump(param_json_data , fp)
            
if __name__ == '__main__':
    if opt.action == 'eval':
        eval(opt)
    elif opt.action == 'param':
        param(opt)
