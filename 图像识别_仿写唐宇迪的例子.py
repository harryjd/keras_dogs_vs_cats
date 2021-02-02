import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image


####################################################################
# 把归一化处理过的图像数据恢复为[0,1]区间的数据，才能显示
def im_convert(tensor):
    # 展示数据
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    image = image * np.array((0.229, 0.224, 0.225)) + \
        np.array((0.485, 0.456, 0.406))
    # 把低于0的值设置为0，超过1的数据设置为1
    image = image.clip(0, 1)
    return image
####################################################################


####################################################################
def set_parameter_requires_grad(a_model, bol_frozen_param):
    if bol_frozen_param:
        for param in a_model.parameters():
            param.requires_grad = False


####################################################################
def initialize_model(model_name, num_classes, bol_frozen_nn_params, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, bol_frozen_nn_params)
        # 再根据
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                    nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, bol_frozen_nn_params)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, bol_frozen_nn_params)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
# data_transforms是一个字典，记录对 [训练数据] 和 [验证数据] 的预处理的 操作
data_transforms = {
    'train': transforms.Compose(
        [transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
         transforms.CenterCrop(224),  # 从中心开始裁剪
         transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
         transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
         # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
         transforms.ColorJitter(
             brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
         transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [
                              0.229, 0.224, 0.225])  # 均值，标准差
         ]),
    'valid': transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])
}

batch_size = 4
# image_datasets是一个字典，分别存放2个数据集的信息，包括图像数据和分类标签
image_datasets = {x: datasets.ImageFolder(os.path.join(
    data_dir, x), data_transforms[x]) for x in ['train', 'valid']}

# 分别为 train 和 valid 两个数据集定义各自的 dataloader
dataloaders = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
# 统计 训练集 和 验证集 的数据量
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

# class_ids是列表，例如：['1', '10', '100', '101', '102', '11', ...]
class_ids = image_datasets['train'].classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# 准备一个数据读取的迭代器
data_iter = iter(dataloaders['valid'])

# region 演示取一个batch的数据，并展示
# fig = plt.figure(figsize=(18, 10))
# columns = 3
# rows = 3


# # 取一个batch_size的数据.
# # 注意:category_ids存储的是类别在image_datasets['train'].classes列表中的序号，不是直接存类别编号
# inputs, category_ids = data_iter.next()

# for idx in range(columns*rows):
#     ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
#     ax.set_title(str(int(class_ids[category_ids[idx]])) + ':' +
#                  cat_to_name[str(int(class_ids[category_ids[idx]]))])
#     plt.imshow(im_convert(inputs[idx]))
# plt.tight_layout()
# plt.show()
# endregion 演示取一个batch的数据，并展示
# 可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']

model_name = 'resnet'
# 是否用人家训练好的特征提取模型来做，也就是沿用别人的权重
bol_frozen_nn_param = True

# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()
my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet152(pretrained=True)
model_ft, input_size = initialize_model(
    model_name, 102, bol_frozen_nn_param, use_pretrained=True)

# GPU计算
model_ft = model_ft.to(my_device)

# 模型保存
filename = 'checkpoint.pth'

# 是否训练所有层
params_to_update = model_ft.parameters()
# print('params_to_update:\n', params_to_update)
# params_to_update = model_ft.named_parameters()
# print('params_to_update:\n', params_to_update)
print("Params to learn:")
if bol_frozen_nn_param:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(
    optimizer_ft, step_size=7, gamma=0.1)  # 学习率每7个epoch衰减成原来的1/10
# 最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()
# 这里不用 criterion = nn.CrossEntropyLoss()


# =====================================================================================================
# =====================================================================================================
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename=filename):
    since = time.time()
    best_acc = 0

    # region 加载模型
    '''
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    '''
    # endregion

    model.to(my_device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()   # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(my_device)
                labels = labels.to(my_device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    # torch.max(outputs, 1)返回每一行的最大值，以及最大值所在的列序号
                    # 预测值为分类在分类列表中的序号，标签值为分类在分类列表中的序号
                    pred_values, pred_idxs = torch.max(outputs, 1)
                    print('outputs:', outputs)
                    print('predict value:', pred_values)
                    print('prdict_category:', pred_idxs)
                    print('labels:', labels.data)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred_idxs == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(
            optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs
# =====================================================================================================


# 训练自定义的最后一层 ———— 全连接层
# model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(
#    model_ft, dataloaders, criterion, optimizer_ft, num_epochs=1, is_inception=(model_name == "inception"))

# 把网络参数再设置为可学习的状态
for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有的参数，学习率调小一点
optimizer = optim.Adam(params_to_update, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.NLLLoss()
# Load the checkpoint

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

#model_ft.class_to_idx = checkpoint['mapping']
# 再次训练，这次训练整个模型
# model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(
#    model_ft, dataloaders, criterion, optimizer, num_epochs=1, is_inception=(model_name == "inception"))

# 得到一个batch的测试数据
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.next()

# 训练完train_datasets之后，model要来测试样本了。在model(test_datasets)之前，需要加上model.eval().
# 否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。
model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

predict_value, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(
    preds_tensor.cpu().numpy())
print(predict_value)
print(preds)
print(labels)

# region 显示验证的图像和分类结果
fig = plt.figure(figsize=(18, 12))
columns = 2
rows = 2

for idx in range(columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} (label:{}/{})".format(cat_to_name[class_ids[int(preds[idx])]],
                                           class_ids[labels[idx].item()], cat_to_name[class_ids[labels[idx].item()]]),
                 color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
plt.tight_layout()
plt.show()
# endregion
