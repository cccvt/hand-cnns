import torch
from torchvision import transforms
import torchvision.models as models

from src.datasets import gtea, gun
from src.options import train_options, error
from src.nets import resnet_adapt, netutils
from src.netscripts import train


opt = train_options.TrainOptions().parse()


# Normalize as imageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Set input tranformations
transformations = ([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

if opt.normalize:
    transformations.append(normalize)
    first_transforms = [transforms.Scale(230), transforms.RandomCrop(224)]
    transformations = first_transforms + transformations

transform = transforms.Compose(transformations)

# Create dataset
if opt.dataset == 'gtea':
    dataset = gtea.GTEA(transform=transform)
    inp_size = dataset.in_channels

print(len(dataset))

dataloader = torch.utils.data.DataLoader(
    dataset, shuffle=True, batch_size=opt.batch_size,
    num_workers=opt.nThreads)

# Load model
resnet = models.resnet18(pretrained=True)
model = resnet_adapt.ResNetAdapt(opt, resnet, dataset.class_nb)

# Load existing weights, opt.continue_training is epoch to load
if opt.continue_training:
    if opt.continue_epoch == 0:
        model.load('latest')
    else:
        model.load(opt.continue_epoch)


if opt.lr != opt.new_lr:
    model_params = model.lr_params()
else:
    model_params = model.parameters()

netutils.print_net(model)

optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr,
                            momentum=opt.momentum)

if opt.criterion == 'MSE':
    criterion = torch.nn.MSELoss()
elif opt.criterion == 'CE':
    criterion = torch.nn.CrossEntropyLoss()

else:
    raise error.ArgumentError(
        '{0} is not among known error functions'.format(opt.criterion))

train.train_net(dataloader, model, optimizer, criterion, opt)
