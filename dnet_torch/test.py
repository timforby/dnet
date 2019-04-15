from models.scasnet import scasnet
import torch
import torch.optim as optim
import torchvision.utils as vutils
from utils.datasets import *
import os
    
img_size = 400
batch_size = 12
lr = 1e-3
total_epochs = 1000
train_paths = ["data/vaihingen/val.txt"]
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

print("Net setup")
net = scasnet(3,8)
"""
    net_opt = torch.optim.Adam([
        {'params' : net.down},
        {'params' : net.cntx},
        {'params' : net.resc},
        {'params' : net.fine, 'lr':lr*20},
        ], lr=lr)
"""

net.cuda()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_paths, patch_size=img_size), batch_size=batch_size, shuffle=False, num_workers=0
)

for epoch in range(total_epochs):
    for batch_i, (path, image_idx, image_total, data) in enumerate(dataloader):
        imagery = data[0]
        net.eval()
        output = net(imagery.cuda())
        

    if epoch % 2 == 0 or epoch == total_epochs-1:
        vutils.save_image(imagery,'%s/%03d_epoch_%s.png' % (output_dir, epoch, "imagery"),normalize=True)
        vutils.save_image(ground_truth,'%s/%03d_epoch_%s.png' % (output_dir, epoch, "ground_truth"),normalize=False)
        segment = torch.argmax(output, dim=1).unsqueeze(1)
        segment = torch.cat([segment%2,(segment//2)%2,(segment//4)%2],dim=1)
        vutils.save_image(segment,'%s/%03d_epoch_%s.png' % (output_dir, epoch, "prediction"),normalize=False)