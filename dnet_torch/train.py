from models.scasnet import scasnet
import torch
import torch.optim as optim
import torchvision.utils as vutils
from utils.datasets import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  

gt_is_bw = True
img_size = 400
batch_size = 4
num_classes = 6#8
lr = 1e-3
total_epochs = 100
train_paths = ["data/test/train.txt", "data/test/gt.txt"]
checkpoint_dir = "checkpoints"
output_dir = "outputs"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print("Net setup")
net = scasnet(3,num_classes)

net_opt = torch.optim.Adam([
    {'params' : net.down},
    {'params' : net.cntx},
    {'params' : net.resc},
    {'params' : net.fine, 'lr':lr*20},
    ], lr=lr)

#net_opt = torch.optim.Adam(net.parameters())
net_cri = torch.nn.CrossEntropyLoss()
net.cuda()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_paths, patch_size=img_size), batch_size=batch_size, shuffle=False, num_workers=0
)

for epoch in range(total_epochs):
    for batch_i, (path, image_idx, image_total, data) in enumerate(dataloader):
        imagery, ground_truth = data
        if not gt_is_bw:
            gt_binary = ground_truth[:,0,:,:]*1+ground_truth[:,1,:,:]*2+ground_truth[:,2,:,:]*4
        else:
            gt_binary = ground_truth[:,0,:,:].squeeze()*(num_classes-1)
        net.zero_grad()
        output = net(imagery.cuda())
        loss = net_cri(output, gt_binary.long().cuda())
        loss.backward()
        net_opt.step()

        print(
            "[Epoch %d/%d, Batch %d/%d, Image %d/%d] [Loss: %.4f]"
            % (
                epoch,
                total_epochs,
                batch_i,
                len(dataloader),
                image_idx[-1]+1,
                image_total[-1],
                loss.item()
            )
        )
        
    if epoch % 50 == 0 or epoch == total_epochs-1:
        torch.save(net.state_dict(),"%s/%d.weights" % (checkpoint_dir, epoch))
    if epoch % 2 == 0 or epoch == total_epochs-1:
        vutils.save_image(imagery,'%s/%03d_epoch_%s.png' % (output_dir, epoch, "imagery"),normalize=True)
        vutils.save_image(ground_truth,'%s/%03d_epoch_%s.png' % (output_dir, epoch, "ground_truth"),normalize=False)
        segment = torch.argmax(output, dim=1).unsqueeze(1)
        if not gt_is_bw:
            segment = torch.cat([segment%2,(segment//2)%2,(segment//4)%2],dim=1)
            vutils.save_image(segment,'%s/%03d_epoch_%s.png' % (output_dir, epoch, "prediction"),normalize=False)
        else:
            segment = segment.float() / (num_classes-1)
            vutils.save_image(segment,'%s/%03d_epoch_%s.png' % (output_dir, epoch, "prediction"),normalize=True,range=(0,1))