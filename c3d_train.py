import cv2
import torch

from src.datasets import something
from src.datasets.gteagazeplusvideo import GTEAGazePlusVideo
from src.datasets.utils import transforms
from src.nets import c3d, c3d_adapt
from src.netscripts import train
from src.options import train_options
from src.utils import evaluation

opt = train_options.TrainOptions().parse()

# Index of sequence item to leave out for validation
leave_out_idx = opt.leave_out


scale_size = (128, 171)
crop_size = (112, 112)
video_transform = transforms.Compose([transforms.Scale(scale_size),
                                      transforms.RandomCrop(crop_size),
                                      transforms.ToTensor()])

# Initialize datasets
leave_out_idx = opt.leave_out

# Initialize dataset
if opt.dataset == 'gteagazeplus':
    all_subjects = ['Ahmad', 'Alireza', 'Carlos',
                    'Rahul', 'Yin', 'Shaghayegh']
    train_seqs, valid_seqs = evaluation.leave_one_out(all_subjects,
                                                      leave_out_idx)
    dataset = GTEAGazePlusVideo(video_transform=video_transform,
                                use_video=False, clip_size=16,
                                original_labels=True,
                                seqs=train_seqs)
    val_dataset = GTEAGazePlusVideo(video_transform=video_transform,
                                    use_video=False, clip_size=16,
                                    original_labels=True,
                                    seqs=valid_seqs)
elif opt.dataset == 'smthgsmthg':
    dataset = something.SomethingSomething(video_transform=video_transform,
                                           clip_size=16, split='train')

    val_dataset = something.SomethingSomething(video_transform=video_transform,
                                               clip_size=16, split='valid')
else:
    raise ValueError('the opt.dataset name provided {0} is not handled\
                     by this script'.format(opt._dataset))

weights = [1/k for k in dataset.class_counts]
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                                                         len(dataset))
# Initialize dataloaders
dataloader = torch.utils.data.DataLoader(
    dataset, shuffle=True, sampler=sampler,
    batch_size=opt.batch_size,
    num_workers=opt.threads, drop_last=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, shuffle=False, batch_size=opt.batch_size,
    num_workers=opt.threads, drop_last=True)

# Initialize neural network
c3dnet = c3d.C3D()
if opt.pretrained:
    c3dnet.load_state_dict(torch.load('data/c3d.pickle'))
model = c3d_adapt.C3DAdapt(opt, c3dnet, dataset.class_nb)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.net.parameters(), lr=0.003)

model.set_criterion(criterion)
model.set_optimizer(optimizer)

train.train_net(dataloader, model, opt,
                valid_dataloader=val_dataloader)
