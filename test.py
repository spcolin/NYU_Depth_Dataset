from NYU_depth_dataset import *
import torchvision

dataset_root='/Users/spcolin/datasets/LargeNYU/train'
file_path='./training_files.txt'

dataset=NYU_Depth(dataset_root,file_path)


rgb,depth=dataset[3]

rgb=torchvision.transforms.ToPILImage()(rgb)
depth=torchvision.transforms.ToPILImage()(depth)

rgb.show()
depth.show()




