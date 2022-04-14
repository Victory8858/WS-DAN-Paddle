"""
- @Author: GaoDing
- @Date: 2022/04/10 20:00
- @Description: 参数配置文件
"""

# Choose Dataset
target_dataset = 'aircraft'  # options: 'aircraft', 'bird', 'car'

# Hyper Parameter
workers = 0  # number of Dataloader workers Windows 不支持多进程
epochs = 80  # number of epochs
batch_size = 6  # batch size, if train, you should set to 12
learning_rate = 0.001  # initial learning rate

# Model
input_size = (448, 448)  # size of training images
image_size = (512, 512)
net_name = 'inception_mixed_6e'  # feature extractor
num_attentions = 32  # number of attention maps
beta = 5e-2  # param for update feature centers

# Model and Logging Save Path
save_dir = "C:/Users/Victory/Desktop/WS-DAN-Paddle-Victory8858/FGVC/" + target_dataset + "/ckpt/"  # Windows Path

# If Checkpoint
model_name = "model.pdparams"
ckpt = False  # if want load model to continue trainning
model_num = 0
