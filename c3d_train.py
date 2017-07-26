import cv2
import torch


from src.datasets import gteagazeplus
from src.nets import c3d
from src.netscripts import train
from src.datasets.utils import transforms
from src.options import train_options

opt = train_options.TrainOptions().parse()

# Index of sequence item to leave out for validation
leave_out_idx = opt.leave_out

model = c3d.C3D(32, opt)

input_size = (112, 112)
video_transform = transforms.Compose([transforms.Scale(input_size),
                                      transforms.ToTensor()])
dataset = gteagazeplus.GTEAGazePlus(video_transform=video_transform,
                                    use_video=False, clip_size=16)

dataloader = torch.utils.data.DataLoader(
    dataset, shuffle=True, batch_size=opt.batch_size,
    num_workers=1, drop_last=True)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

model.set_criterion(criterion)
model.set_optimizer(optimizer)

train.train_net(dataloader, model, criterion, opt, optimizer,
                valid_dataloader=None)
