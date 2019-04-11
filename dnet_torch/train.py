from models.scasnet import scasnet
import torch
import torch.optim as optim
import torchvision.utils as vutils
from utils.datasets import *

    
img_size = 400
batch_size = 4
lr = 1e-3
total_epochs = 100
train_paths = ["data/vaihingen/train.txt", "data/vaihingen/ground_truth.txt"]
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

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
    for batch_i, (path, patch_idx, patch_total, data) in enumerate(dataloader):
        imagery, ground_truth = data
        
        gt_binary = ground_truth[:,0,:,:]*1+ground_truth[:,1,:,:]*2+ground_truth[:,2,:,:]*4
        """
            net.zero_grad()
            output = net(imagery)
            loss = net_cri(output, gt_binary.long())
            loss.backward()
            optimizer.step()
        """
        print(
            "[Epoch %d/%d, Image %d/%d, Batch %d/%d] [Loss: %.4f]"
            % (
                epoch,
                total_epochs,
                batch_i,
                len(dataloader),
                patch_idx[batch_size-1],
                patch_total[batch_size-1],
                65#loss.item()
            )
        )
        
    if epoch % 2 == 0:
        model.save_weights("%s/%d.weights" % (checkpoint_dir, epoch))