import os
from model import attribute_transformer as ats
from model import vision_transformer as vits
# -----------------
# DATASET ROOTS
# -----------------
cwd = os.getcwd()

server_path = "/db/pszzz/" 

dataset_root = os.path.join(server_path,'NCD_dataset')
cifar_10_root = os.path.join(server_path,'NCD_dataset/cifar10')
cifar_100_root = os.path.join(server_path,'NCD_dataset/cifar100')
cub_root = os.path.join(server_path,'NCD_dataset/cub')
aircraft_root = os.path.join(server_path,'NCD_dataset/aircraft/fgvc-aircraft-2013b')
herbarium_dataroot = os.path.join(server_path,'NCD_dataset/herbarium_19/')
imagenet_root = imagenet_root = os.path.join(server_path,'NCD_dataset/imagenet/') #"/nfs/data_chaos/datasets/ImageNet"


cars_root = os.path.join(server_path,'NCD_dataset/scars')
pets_root = os.path.join(server_path,'NCD_dataset/pets')
flower_root = os.path.join(server_path,'NCD_dataset/flower102')
food_root = os.path.join(server_path,'NCD_dataset/food-101')


# OSR Split dir
osr_split_dir = os.path.join(server_path,'NCD_dataset/ssb_splits')
# -----------------
# PRETRAIN PATHS
# -----------------

# -----------------
dino_pretrain_path = os.path.join(server_path,'NCD_dataset/dino/dino_vitbase16_pretrain.pth')
dino_base_pretrain_path = os.path.join(server_path,'NCD_dataset/dino/dino_vitbase16_pretrain.pth')
dino_small_pretrain_path = os.path.join(server_path,'NCD_dataset/dino/dino_deitsmall16_pretrain.pth')
moco_pretrain_path = os.path.join(server_path,'NCD_dataset/moco/vit-b-300ep.pth.tar')
mae_pretrain_path = os.path.join(server_path,'NCD_dataset/mae/mae_pretrain_vit_base.pth')

# Dataset partitioning paths
km_label_path = os.path.join(server_path,'NCD_dataset/attribute_out/partition_out/km_labels')
subset_len_path = os.path.join(server_path,'NCD_dataset/attribute_out/partition_out/subset_len')

# -----------------
# OTHER PATHS
# -----------------
feature_extract_dir = os.path.join(server_path,'NCD_dataset/attribute_out/extracted_features')     # Extract features to this directory
exp_root = os.path.join(server_path,'NCD_dataset/attribute_out/logs')          # All logs and checkpoints will be saved here
