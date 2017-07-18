import torch
from torchvision import transforms
import torchvision.models as models

from src.datasets import gtea, gun
from src.options import train_options, error
from src.nets import resnet_adapt, netutils
from src.netscripts import train
from src.utils.normalize import Unnormalize


opt = train_options.TrainOptions().parse()


# Normalize as imageNet
img_means = [0.485, 0.456, 0.406]
img_stds = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=img_means,
                                 std=img_stds)

# Compute reverse of normalize transform
unnormalize = Unnormalize(mean=img_means, std=img_stds)

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
    dataset = gtea.GTEA(transform=transform, untransform=unnormalize,
                        seqs=['S2', 'S3', 'S4'])
    valid_dataset = gtea.GTEA(transform=transform,
                              untransform=unnormalize,
                              seqs=['S1'])
    valid = True

elif opt.dataset == 'gun':
    test_subject_id = 1
    # Leave one out training
    seqs = ['Subject1', 'Subject2',
            'Subject3', 'Subject4',
            'Subject5', 'Subject6',
            'Subject7', 'Subject8']
    valid_seqs = [seqs.pop(test_subject_id)]
    train_seqs = seqs

    dataset = gun.GUN(transform=transform, untransform=unnormalize,
                      seqs=train_seqs)

    valid_dataset = gun.GUN(transform=transform, untransform=unnormalize,
                            seqs=valid_seqs)
    valid = True

print('Dataset size : {0}'.format(len(dataset)))

dataloader = torch.utils.data.DataLoader(
    dataset, shuffle=True, batch_size=opt.batch_size,
    num_workers=opt.threads, drop_last=True)

if valid:
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   shuffle=False,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.threads)

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

model.set_criterion(criterion)
model.set_optimizer(optimizer)

train.train_net(dataloader, model, criterion, opt, optimizer,
                valid_dataloader=valid_dataloader)
