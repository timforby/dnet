from models.scasnet import scasnet
import torch
import torch.optim as optim
import torchvision.utils as vutils
from utils.datasets import *
import os
    
img_size = 400
batch_size = 12
lr = 1e-3
gt_is_bw = False
total_epochs = 1000
train_paths = ["data/vaihingen/val.txt"]
result_dir = "results"
model_dir = "checkpoints/999.weights"
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
net.load_state_dict(torch.load(model_dir))
net.cuda()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_paths, patches=False), batch_size=1, shuffle=False, num_workers=0
)

for image_i, (path, image_idx, image_total, data) in enumerate(dataloader):
    imagery = data[0]
    net.eval()
    width = np.arange(0, imagery.shape[2], img_size)[:-1]
    heigth = np.arange(0, imagery.shape[3], img_size)[:-1]
    output = torch.zeros((8,len(width)*img_size,len(heigth)*img_size))
    for x in width:
        for y in heigth:
            patch = imagery[0,:,x:x+img_size,y:y+img_size].unsqueeze(0)
            out = net(patch.float().cuda()).detach().cpu()
            output[:,x:x+img_size,y:y+img_size] = out[0]

    segment = np.argmax(output, axis=0).unsqueeze(0)
    if not gt_is_bw:
        segment = torch.cat([segment%2,(segment//2)%2,(segment//4)%2],dim=0)
        vutils.save_image(segment,'%s/%03d_%s.png' % (result_dir,image_i, "test"),normalize=False)
    else:
        segment = segment.float() / (num_classes-1)
        vutils.save_image(segment,'%s/%03d_%s.png' % (result_dir,image_i, "test"),normalize=True,range=(0,1))