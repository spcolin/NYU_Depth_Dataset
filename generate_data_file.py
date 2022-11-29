import os

dataset_root='/Users/spcolin/datasets/LargeNYU/train'


def generate_file(dataset_root,save_path='./training_files.txt'):

    training_data_list=[]

    scene_folder=os.listdir(dataset_root)

    scene_folder.remove('.DS_Store')

    for scene in scene_folder:

        rgb_folder=os.path.join(dataset_root,scene,'rgb')
        depth_folder=os.path.join(dataset_root,scene,'depth')

        rgb_files=os.listdir(rgb_folder)
        depth_files=os.listdir(depth_folder)

        for i in depth_files:

            if i in rgb_files:

                data_str=os.path.join(scene,'rgb',i)+' '+os.path.join(scene,'depth',i)

                training_data_list.append(data_str)

        # break


    print("total training data:",len(training_data_list))

    file=open(save_path,'w')


    for line in training_data_list:

        file.write(line)
        file.write('\n')

    file.close()

    print('file saved at:',save_path)



generate_file(dataset_root)



