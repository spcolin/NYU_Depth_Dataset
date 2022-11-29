from torch.utils.data import Dataset
from PIL import Image
import os,random
import numpy as np
from torchvision import transforms

class NYU_Depth(Dataset):

    def __init__(self,dataset_root,file_path,target_height=480,target_width=640,
                mode='train',keep_border=True,random_crop=True,random_rotate=True,
                rotate_degree=2.5,depth_scale=6553.5,random_flip=True,color_aug=True):
        super().__init__()

        """
        dataset_root:the path placing NYU depth dataset
        file_path:the path of coupled rgb and depth map
        dataset_root+file_path locates the training data
        """

        self.dataset_root=dataset_root
        self.target_height=target_height
        self.target_width=target_width
        self.mode=mode
        self.keep_border=keep_border
        self.random_crop=random_crop
        self.random_rotate=random_rotate
        self.rotate_degree=rotate_degree
        self.depth_scale=depth_scale
        self.random_flip=random_flip
        self.color_aug=color_aug
        
        file=open(file_path)

        self.files=file.readlines()

        file.close()

    def __len__(self):

        return len(self.files)

    def __getitem__(self,index):

        file_path=self.files[index]
        file_path=file_path.strip('\n')

        rgb_path,depth_path=file_path.split(' ')
        
        rgb_path=os.path.join(self.dataset_root,rgb_path)
        depth_path=os.path.join(self.dataset_root,depth_path)

        rgb=Image.open(rgb_path)
        depth=Image.open(depth_path)

        if self.mode=='train':

            if self.keep_border:
                depth = np.array(depth)
                valid_mask = np.zeros_like(depth)
                valid_mask[45:472, 43:608] = 1
                depth[valid_mask==0] = 0
                depth = Image.fromarray(depth)
            else:
                rgb = rgb.crop((43, 45, 608, 472))
                depth = depth.crop((43, 45, 608, 472))

            if self.random_crop:
                rgb,depth=self.crop_resize(rgb,depth,self.target_height,self.target_width)
            
            if self.random_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.rotate_degree
                rgb = self.rotate_image(rgb, random_angle)
                depth = self.rotate_image(depth, random_angle, flag=Image.NEAREST)

            if self.random_flip:
                rgb,depth=self.flip_image(rgb,depth)

            if self.color_aug:
                rgb=self.augment_image(rgb)

            to_tensor_and_norm=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            rgb_tensor=to_tensor_and_norm(rgb)
            depth_tensor=transforms.ToTensor()(depth)

            depth_tensor=depth_tensor/self.depth_scale

            return rgb_tensor,depth_tensor

        else:

            to_tensor_and_norm=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            rgb_tensor=to_tensor_and_norm(rgb)
            depth_tensor=transforms.ToTensor()(depth)

            depth_tensor=depth_tensor/self.depth_scale

            return rgb_tensor,depth_tensor

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def flip_image(self,rgb,depth):

        do_flip = random.random()
        if do_flip > 0.5:
            # print('flip image')
            rgb=rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth=depth.transpose(Image.FLIP_LEFT_RIGHT)

        return rgb,depth

    def crop_resize(self, rgb, depth, target_height, target_width):

        full_width,full_height=rgb.size

        height_scale=random.uniform(0.8,1)
        width_scale=random.uniform(0.8,1)

        height=int(full_height*height_scale)
        width=int(full_width*width_scale)

        x = random.randint(0, full_width - width)
        y = random.randint(0, full_height - height)

        rgb=rgb.crop((x,y,x+width,y+height))
        depth=depth.crop((x,y,x+width,y+height))

        rgb=rgb.resize((target_width,target_height),Image.ANTIALIAS)
        depth=depth.resize((target_width,target_height),Image.ANTIALIAS)

        return rgb,depth

    def augment_image(self, rgb):
        
        color_transform=transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1)

        rgb=color_transform(rgb)

        return rgb




