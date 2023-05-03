import glob
import pandas as pd
import os
from PIL import Image
import numpy as np
from random import sample
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboard import program
import webbrowser
from torch.utils.tensorboard import SummaryWriter
import argparse
from ResNet_sz56 import resnet142
# from VGG import MiniVGG19_sz56
# import torchvision.models as models

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, help='model name')
parser.add_argument('-i','--image_folder', help='image folder', required=True)
parser.add_argument('-c','--cell_table', help='cell table', required=True)
parser.add_argument('-o','--output_folder', help='output folder', required=True)
parser.add_argument("-p", "--pixel_size", type=float, help="size of pixels in micron", required=True)
parser.add_argument("-bz", "--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("-ne", "--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.00001, help="learning rate")
parser.add_argument("-pr", "--checkpoint_pass_rate", type=float, default=0.0, help="checkpoint pass rate")
parser.add_argument("-tb", "--tensorboard", action='store_true', help="tensorboard")
parser.add_argument("-tp", "--tensorboard_port", type=str, default="6006", help="tensorboard port")
opt = parser.parse_args()
print(opt)

if opt.tensorboard:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', os.path.join(opt.output_folder, "runs"), '--port', opt.tensorboard_port])
        url = tb.launch()
        print(f"Tensorboard listening on {url}")
        webbrowser.open(url)
        tensorboard_writer = SummaryWriter(os.path.join(opt.output_folder, "runs"))
        
# GPU environment by CUDA
# cuda = True if torch.cuda.is_available() else False

# # Tensor types
# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor 

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
            # T.CenterCrop(image_size),
            # T.Resize(image_size, antialias=True),
            # T.Resize(224, antialias=True),
            T.Normalize(image_mean, image_std),     
            T.RandomRotation(degrees=(0, 90)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            ]

        self.transform = T.Compose(transformObjectList)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_data)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        # uuid = self.image_data.iloc[index]['Object ID']
        label = self.image_data.iloc[index]['label']
        imageFile = self.image_data.iloc[index]['image_filepath']

        img = Image.open(imageFile)
        img = self.transform(img)
        
        return (img, label)
    
    def __partition(self, lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def train(opt):
    # Prepare image data
    image_filepath_list = []
    image_uuid_list = []
    image_filename_list = glob.glob(os.path.join(opt.image_folder, '*','*'))
    
    # Normalization 
    sampled_image_filename_list = sample(image_filename_list,100)
    sampled_image_arraylist = []
    
    image_fileext = None
    image_size = None
    
    for image_filename in sampled_image_filename_list:
        img_buf = Image.open(image_filename)
        img_ary = np.array(img_buf).astype(np.float32)
        
        assert img_ary.shape[0] == img_ary.shape[1]
        assert image_size == None or image_size == img_ary.shape[0]
        image_size = img_ary.shape[0]
        
        sampled_image_arraylist.append(img_ary.reshape((1, *img_ary.shape)))
    
        _, filename = os.path.split(image_filename)
        _, fileext = os.path.splitext(image_filename)
        
        assert image_fileext == None or image_fileext == fileext
        image_fileext = fileext
        
    sampled_image_arraylist_concat = np.concatenate(sampled_image_arraylist)
    image_mean = sampled_image_arraylist_concat.mean(axis=(0,1,2))
    image_std = (sampled_image_arraylist_concat-image_mean).std(axis=(0,1,2))
    
    for image_filename in image_filename_list:
        _, filename = os.path.split(image_filename)
        image_uuid, _ = os.path.splitext(filename)
    
        image_filepath_list.append(image_filename)
        image_uuid_list.append(image_uuid)
        
    image_df = pd.DataFrame({'image_filepath': image_filepath_list, 'Object ID':image_uuid_list})
    image_df.set_index('Object ID', inplace=True)

    cell_df = pd.read_table(opt.cell_table)
    cell_df.set_index('Object ID', inplace=True)
    
    data_df = cell_df.join(image_df)
    data_df = data_df[['image_filepath', 'Class']]
    data_df.dropna(subset=['Class'], inplace=True)
    data_df['label'] = data_df.Class.apply(lambda x: int(x.replace('xenium: cluster: ', ''))-1)
    n_classes = len(data_df['label'].unique())
    
    # Curate image and transcript data by cell UUID
    total_uuid_list = [value for value in data_df.index]
    
    # Split all data into training, validation and test uuids.
    training_uuid_list = total_uuid_list[:int(len(total_uuid_list)*0.9)]
    validation_uuid_list = total_uuid_list[int(len(total_uuid_list)*0.9):]
    
    # Create image data according to training, validation and test uuids.
    training_df = data_df.reindex(training_uuid_list)
    validation_df = data_df.reindex(validation_uuid_list)
    
    # Create datasets
    training_dataset = SingleCellImageDataset(
        training_df, 
        image_size=image_size,
        image_mean=image_mean, 
        image_std=image_std) 
    
    validation_dataset = SingleCellImageDataset(
        validation_df,  
        image_size=image_size,
        image_mean=image_mean, 
        image_std=image_std) 

    # Create dataloaders
    train_dataloader = DataLoader(training_dataset, opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(validation_dataset, opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    param_json_data = {
                "parameters": vars(opt),
                "image_size": image_size,
                "image_std": image_std.tolist(),
                "image_mean": image_mean.tolist(),
                "pixel_size": opt.pixel_size,
                "n_classes": n_classes
                }
          
    # net = models.resnet18().to('cuda')
    # net = models.alexnet().to('cuda')
    # net = models.vgg16().to('cuda')
    # net = models.squeezenet1_0().to('cuda')
    # net = models.densenet161().to('cuda')
    # net = models.inception_v3().to('cuda')
    # net = models.googlenet().to('cuda')
    # net = models.shufflenet_v2_x1_0().to('cuda')
    # net = models.mobilenet_v2() .to('cuda')
    
    net = resnet142(n_classes).to('cuda')
    # net = MiniVGG19_sz56(n_classes).to('cuda')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = opt.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
    
    if opt.tensorboard:
        inp = next(iter(train_dataloader))
        inputs, labels = inp
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        tensorboard_writer.add_graph(net, inputs)
    
    last_cls_acc = 0
        
    for epoch in tqdm(range(opt.n_epochs)):
        losses = []
        # running_loss = 0
        net.train()
        for inp in train_dataloader:
            inputs, labels = inp
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
        
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
    
            loss.backward()
            optimizer.step()
            
            if opt.tensorboard:
                tensorboard_writer.add_scalar("train_loss", loss.item(), epoch)
                tensorboard_writer.flush()
    
        avg_loss = sum(losses)/len(losses)
        scheduler.step(avg_loss)
    
        correct = 0
        total = 0
        
        net.eval()
        with torch.no_grad():
            for data in val_dataloader:
                images, labels = data
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = net(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_cls_acc = correct/total
        
        if opt.tensorboard:
            tensorboard_writer.add_scalar("val_acc", 100*val_cls_acc, epoch)
   
        if val_cls_acc >= last_cls_acc*opt.checkpoint_pass_rate:
            last_cls_acc = val_cls_acc
            torch.save({'model_state': net.state_dict(),
                        'parameters': param_json_data
                        }, os.path.join(opt.output_folder, opt.model_name+".ckpt"))
        else:
            checkpoint = torch.load(os.path.join(opt.output_folder, opt.model_name+".ckpt"))
            net.load_state_dict(checkpoint['model_state'])
                
    torch.save({'model_state': net.state_dict(),
                'parameters': param_json_data
                }, os.path.join(opt.output_folder, opt.model_name+".pt"))
              
    input("Training Done! Press Enter to continue...")
    
if __name__ == '__main__':
    train(opt)
