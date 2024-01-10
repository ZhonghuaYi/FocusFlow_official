import os


# 读取文件夹下所有文件的文件名
def get_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

file_dir = "F:\FlowDataset\KITTI-custom\\training\image_2"

# 读取文件夹下所有文件的文件名并输出到KITTI_split.txt中
with open("KITTI_split.txt", "w") as f:
    for file_name in get_file_name(file_dir):
        f.write(file_name + "\n")
