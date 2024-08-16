import argparse
import sys
import os
import random

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

parser = argparse.ArgumentParser(description='ensemble')
parser.add_argument('--id_sp', type=float, default=0.9)
parser.add_argument('--p', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_iters', type=int, default=5001)
parser.add_argument('--n_restarts', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--colors', type=int, default=32)

args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print("Loading dataset...")

mnist = datasets.MNIST('mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])
images_train = mnist_train[0] 
images_train = images_train.reshape((-1, 28, 28))[:, ::2, ::2]
labels_train = mnist_train[1]

def duplicate_images(images_):
    images_ = torch.stack([images_, images_, images_], dim=1)

    images_1 = torch.cat([images_, images_, images_], dim=2)
    images_2 = torch.cat([images_1, images_1, images_1], dim=3)
    images_2[:, :, :14, :14] = 0
    images_2[:, :, :14, 14:28] = 0
    images_2[:, :, :14, 28:42] = 0

    images_2[:, :, 14:28, :14] = 0
    # mid leave blank
    images_2[:, :, 14:28, 28:42] = 0

    images_2[:, :, 28:42, :14] = 0
    images_2[:, :, 28:42, 14:28] = 0
    images_2[:, :, 28:42, 28:42] = 0
    return images_2

dup_images_train = duplicate_images(images_train)


mnist_val = (mnist.data[50000:], mnist.targets[50000:])
images_val = mnist_val[0] 
images_val = images_val.reshape((-1, 28, 28))[:, ::2, ::2]
labels_val = mnist_val[1]

dup_images_val = duplicate_images(images_val)

# Generate Meta List

color_names = [
    "Red",
    "Green",
    "Blue",
    "Yellow",
    "Cyan",
    "Magenta",
    "Black",
    "White",
    "Gray",
    "Orange"
]

colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 0, 0),      # Black
    (255, 255, 255),# White
    (128, 128, 128),# Gray
    (255, 165, 0),  # Orange
]

def get_loss(preds, labels):
    one_hot_labels = F.one_hot(labels, num_classes=10)
    loss = ((one_hot_labels - preds)**2).mean()
    return loss

#top_left 
# blocks 
x_list = [(0), (0),  (0),  (14), (14), (28), (28), (28),]
y_list = [(0), (14), (28), (0),  (28), (0),  (14), (28),]

meta_locs = []
for i in range(len(x_list)):
    xrange = x_list[i]
    yrange = y_list[i]
    import random
    for iii in range(2):
        for jjj in range(2):
            x_begin = xrange+iii*7
            x_end = xrange+iii*7 + 7
            y_begin = yrange+jjj*7
            y_end = yrange+jjj*7  + 7
            
            relocs = list(range(len(colors)))

            # Shuffle the list randomly
            random.shuffle(relocs)
            color_names_relocs = [color_names[i] for i in relocs]
            color_relocs = [colors[i] for i in relocs]
            color_dict_relocs = dict(zip(list(range(len(color_relocs))), color_relocs))
            meta_locs.append(
                {
                    "xrange": xrange,
                    "yrange": yrange,
                    "iii": iii,
                    "jjj": jjj,
                    "relocs": relocs,
                    "color_names_relocs": color_names_relocs,
                    "color_relocs": color_relocs,
                    "color_dict_relocs": color_dict_relocs,
                })


# Spurious Feature Function
# spurious_correlation = 0.9

def generate_spurious_feature(meta_locs_in, spurious_correlation, images_in, label_):
    images_ = images_in.clone()
    color_correlation_list = {}
    for i_ in range(len(meta_locs_in)):
        color_correlation_list[i_] = []
        imeta = meta_locs_in[i_]
        xrange = imeta["xrange"]
        yrange = imeta["yrange"]
        iii = imeta["iii"]
        jjj = imeta["jjj"]
        color_relocs = imeta["color_relocs"]
        color_names_relocs = imeta["color_names_relocs"]
#         print("processing", xrange, yrange, iii, jjj)
        x_begin = xrange+iii*7
        x_end = xrange+iii*7 + 7
        y_begin = yrange+jjj*7
        y_end = yrange+jjj*7  + 7


        def reshape_color(color_tensor, dim=0):
            return torch.unsqueeze(torch.unsqueeze(color_tensor[:, dim], 1), 2).repeat(1, 7, 7)
        for _i_n in range(len(images_)):
            if random.random() < spurious_correlation:
                label_int = int(label_[_i_n].item())
                this_color = color_relocs[label_int]

                color_correlation = 1
            else:
                this_color = random.choice(colors)
                color_correlation = 0

#             color_correlation_list.append(color_correlation)
            images_[_i_n, 0, x_begin:x_end, y_begin:y_end] = this_color[0]
            images_[_i_n, 1, x_begin:x_end, y_begin:y_end] = this_color[1]
            images_[_i_n, 2, x_begin:x_end, y_begin:y_end] = this_color[2]
            color_correlation_list[i_].append(color_correlation)
    return {"images": images_,
            "labels": label_, 
            "color_correlation_list": color_correlation_list}
 


# Spurious Feature Function
# spurious_correlation = 0.9

def generate_spurious_feature_single_color(meta_locs_in, spurious_correlation, images_in, label_):
    images_ = images_in.clone()
    color_correlation_list = {}
    
    def reshape_color(color_tensor, dim=0):
        return torch.unsqueeze(torch.unsqueeze(color_tensor[:, dim], 1), 2).repeat(1, 7, 7)
    for _i_n in range(len(images_)):
        color_relocs = meta_locs_in[0]["color_relocs"]
        color_names_relocs = meta_locs_in[0]["color_names_relocs"]
        if random.random() < spurious_correlation:
            label_int = int(label_[_i_n].item())
            this_color = color_relocs[label_int]

            color_correlation = 1
        else:
            this_color = random.choice(colors)
            color_correlation = 0
        for i_ in range(len(meta_locs_in)):
            color_correlation_list[i_] = []
            imeta = meta_locs_in[i_]
            xrange = imeta["xrange"]
            yrange = imeta["yrange"]
            iii = imeta["iii"]
            jjj = imeta["jjj"]
    #         print("processing", xrange, yrange, iii, jjj)
            x_begin = xrange+iii*7
            x_end = xrange+iii*7 + 7
            y_begin = yrange+jjj*7
            y_end = yrange+jjj*7  + 7


#             color_correlation_list.append(color_correlation)
            images_[_i_n, 0, x_begin:x_end, y_begin:y_end] = this_color[0]
            images_[_i_n, 1, x_begin:x_end, y_begin:y_end] = this_color[1]
            images_[_i_n, 2, x_begin:x_end, y_begin:y_end] = this_color[2]
            color_correlation_list[i_].append(color_correlation)
    return {"images": images_,
            "labels": label_, 
            "color_correlation_list": color_correlation_list}


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.to(torch.float32)/255
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
print("processing training data")
assert args.colors in [1, 32]
if args.colors == 32:
    gen_fun = generate_spurious_feature
else:
    gen_fun = generate_spurious_feature_single_color

results_train = gen_fun(
    meta_locs, args.id_sp,
    dup_images_train,
    labels_train)
print("processing IID validation data")
results_valIID = gen_fun(
    meta_locs, args.id_sp,
    dup_images_val,
    labels_val)
print("processing OOD validation data")
results_valOOD = gen_fun(
    meta_locs, 1 - args.p,
    dup_images_val,
    labels_val)


train_ds = CustomImageDataset(images=results_train["images"], 
                   labels=results_train["labels"])
#Test
val_ds_ood = CustomImageDataset(images=results_valOOD["images"], 
                   labels=results_valOOD["labels"])
val_ds_iid = CustomImageDataset(images=results_valIID["images"], 
                   labels=results_valIID["labels"])
# IID Test

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader_ood = DataLoader(val_ds_ood, batch_size=64, shuffle=True)
val_loader_iid = DataLoader(val_ds_iid, batch_size=64, shuffle=True)

print("Done")

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
batch_size = 100
n_iters = 5001
epochs = n_iters / (len(train_loader))
input_dim = len(results_train["images"][0].view(-1))
output_dim = 10
lr_rate = 0.001


# train_dataset = datasets.MNIST('~/datasets/mnist', train=True, download=True, transform=transforms.ToTensor())
# test_dataset = datasets.MNIST('~/datasets/mnist', train=False, download=True, transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim))

    def forward(self, x):
        outputs = self.model(x)
        return outputs


results = []

for restart in range(args.n_restarts):
    print("-" * 20, F"restart {restart}", "-" * 20)
    if restart >= args.n_restarts // 2:
        model2 = MLP(input_dim, output_dim).cuda()
        model1 = MLP(input_dim, output_dim).cuda()
    else:
        model1 = MLP(input_dim, output_dim).cuda()
        model2 = MLP(input_dim, output_dim).cuda()

    criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr_rate)


    def evaluate_acc(evaluation_loader, model_):
        total = 0
        correct = 0          
        evaluation_iter = evaluation_loader.__iter__()
        for j in range(len(evaluation_loader)):
            images, labels = evaluation_iter.__next__()
            images, labels = images.cuda(), labels.cuda()
            images = images.view(len(images), -1)
            outputs = model_(images)
            _, predicted = torch.max(outputs.data, 1)
            total+= labels.size(0)
            correct+= (predicted == labels).sum()
        accuracy = 100 * correct/total
        return accuracy

    def evaluate_acc_ensemble(evaluation_loader, model1_, model2_):
        total = 0
        correct = 0          
        evaluation_iter = evaluation_loader.__iter__()
        for j in range(len(evaluation_loader)):
            images, labels = evaluation_iter.__next__()
            images, labels = images.cuda(), labels.cuda()
            images = images.view(len(images), -1)
            outputs = (model1_(images) + model2_(images))/2
            _, predicted = torch.max(outputs.data, 1)
            total+= labels.size(0)
            correct+= (predicted == labels).sum()
        accuracy = 100 * correct/total
        return accuracy

    iter = 0
    for epoch in range(int(epochs)):
        train_iter = train_loader.__iter__()
        for i in range(len(train_loader)):
            images, labels = train_iter.__next__()
            images, labels = images.cuda(), labels.cuda()
            images = (images.view(len(images), -1))
            labels = (labels)
            outputs2 = model2(images)
            loss2 = get_loss(outputs2, labels)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            outputs1 = model1(images)
            loss1 = get_loss(outputs1, labels)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            if iter%500==0:

                # calculate Accuracy
                acc_ood1 = evaluate_acc(evaluation_loader=val_loader_ood, model_=model1)
                acc_iid1 = evaluate_acc(evaluation_loader=val_loader_iid, model_=model1)
                acc_ood2 = evaluate_acc(evaluation_loader=val_loader_ood, model_=model2)
                acc_iid2 = evaluate_acc(evaluation_loader=val_loader_iid, model_=model2)
                ass_ensemble_ood  = evaluate_acc_ensemble(val_loader_ood, model1, model2)
                print(F"It: {iter}, Loss: {loss1.item():.4f}, IID Acc: {acc_iid1:.2f} and {acc_iid2:.2f}, OOD Acc: {acc_ood1:.2f} and {acc_ood2:.2f}, Ensemble OOD:{ass_ensemble_ood:.4f}.")
                res = {
                    "restart": restart,
                    "iter": iter,
                    "loss": loss1.item(),
                    "acc_ood1": acc_ood1.item(),
                    "acc_iid1": acc_iid1.item(),
                    "acc_ood2": acc_ood2.item(),
                    "acc_iid2": acc_iid2.item(),
                    "ass_ensemble_ood": ass_ensemble_ood.item(),
                }
                results.append(res)
            iter += 1
import pandas as pd
df_res = pd.DataFrame(results)
df_res = df_res[df_res.iter == 4500] # report the last step performance
print(F"Model1 OOD Acc={df_res.acc_ood1.mean():.4f}", 
    F"Model2 OOD Acc={df_res.acc_ood2.mean():.4f}",
    F"ModelEnsemble OOD Acc={df_res.ass_ensemble_ood.mean():.4f}",)
