from models.scasnet import scasnet
import torch
import torch.optim as optim
import torchvision.utils as vutils
from utils.datasets import *
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
    
img_size = 400
batch_size = 4
lr = 1e-3
total_epochs = 100
train_paths = ["data/vaihingen/train.txt", "data/vaihingen/target.txt"]
checkpoint_dir = "checkpoints"
output_dir = "outputs"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print("Net setup")
net = scasnet(3,8)
'''
net_opt = torch.optim.Adam([
    {'params' : net.down},
    {'params' : net.cntx},
    {'params' : net.resc},
    {'params' : net.fine, 'lr':lr*20},
    ], lr=lr)
'''
net_opt = torch.optim.Adam(net.parameters())
net_cri = torch.nn.CrossEntropyLoss()
net.cuda()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_paths, patch_size=img_size), batch_size=batch_size, shuffle=False, num_workers=0
)

for epoch in range(total_epochs):
    for batch_i, (path, image_idx, image_total, data) in enumerate(dataloader):
        imagery, ground_truth = data
        gt_binary = ground_truth[:,0,:,:]*1+ground_truth[:,1,:,:]*2+ground_truth[:,2,:,:]*4

        net.zero_grad()
        output = net(imagery.cuda())
        loss = net_cri(output, gt_binary.long().cuda())
        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d, Image %d/%d] [Loss: %.4f]"
            % (
                epoch,
                total_epochs,
                batch_i,
                len(dataloader),
                image_idx[batch_size-1]+1,
                image_total[batch_size-1],
                loss.item()
            )
        )
        
    if epoch % 2 == 0:
        net.save_weights("%s/%d.weights" % (checkpoint_dir, epoch))
        vutils.save_image(imagery,'%s/%03d_epoch_%s.png' % (output_dir, epoch, "imagery"),normalize=True)
        vutils.save_image(ground_truth,'%s/%03d_epoch_%s.png' % (output_dir, epoch, "ground_truth"),normalize=False)
        segment = torch.argmax(output, dim=1).reshape((1, output.shape[0], output.shape[1]))
        segment = torch.cat([segment%2,(segment//2)%2,(segment//4)%2],dim=0)
        vutils.save_image(segment,'%s/%03d_epoch_%s.png' % (output_dir, epoch, "prediction"),normalize=False)