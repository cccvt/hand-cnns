import cv2
import torch
from torchvision import transforms
import torchvision.models as models

from src.datasets import gtea, gun
from src.datasets.gteagazeplusimage import GTEAGazePlusImage
from src.datasets.smthgimage import SmthgImage
from src.options import train_options, error
from src.nets import resnet_adapt
from src.netscripts import train
from src.utils.normalize import Unnormalize
from src.utils import evaluation


def run_training(opt):
    # Normalize as imageNet
    img_means = [0.485, 0.456, 0.406]
    img_stds = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=img_means,
                                     std=img_stds)

    # Compute reverse of normalize transfor
    unnormalize = Unnormalize(mean=img_means, std=img_stds)

    # Set input tranformations
    transformations = ([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    final_size = 224
    if opt.normalize:
        transformations.append(normalize)
    first_transforms = [transforms.Scale(230),
                        transforms.RandomCrop(final_size)]
    transformations = first_transforms + transformations

    transform = transforms.Compose(transformations)
    base_transform = transforms.Compose([
        transforms.Scale(final_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Index of sequence item to leave out for validation
    leave_out_idx = opt.leave_out

    if opt.dataset == 'gteagazeplus':
        # Create dataset
        valid = True
        all_subjects = ['Ahmad', 'Alireza', 'Carlos',
                        'Rahul', 'Yin', 'Shaghayegh']
        train_seqs, valid_seqs = evaluation.leave_one_out(all_subjects,
                                                          leave_out_idx)
        dataset = GTEAGazePlusImage(transform=transform,
                                    untransform=unnormalize,
                                    seqs=train_seqs)
        valid_dataset = GTEAGazePlusImage(transform=transform,
                                          base_transform=base_transform,
                                          untransform=unnormalize,
                                          seqs=valid_seqs)
        valid = True

    elif opt.dataset == 'gun':
        test_subject_id = 2
        # Leave one out training
        seqs = ['Subject1', 'Subject2',
                'Subject3', 'Subject4',
                'Subject5', 'Subject6',
                'Subject7', 'Subject8']
        train_seqs, valid_seqs = evaluation.leave_one_out(seqs,
                                                          test_subject_id)

        dataset = gun.GUN(transform=transform, untransform=unnormalize,
                          seqs=train_seqs)

        valid_dataset = gun.GUN(transform=transform, untransform=unnormalize,
                                seqs=valid_seqs)
        valid = True

    elif opt.dataset == 'smthgsmthg':
        dataset = SmthgImage(split='train', transform=transform,
                             untransform=unnormalize, base_transform=base_transform)
        valid_dataset = SmthgImage(
            split='valid', transform=transform,
            untransform=unnormalize,
            base_transform=base_transform)
        valid = True

    print('Dataset size : {0}'.format(len(dataset)))

    # Initialize sampler
    if opt.weighted_training:
        weights = [1 / k for k in dataset.class_counts]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                                                                 len(dataset))
    else:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)

    # Initialize dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size,
        num_workers=opt.threads, sampler=sampler)

    if valid:
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, shuffle=False,
            batch_size=opt.batch_size,
            num_workers=opt.threads)

    # Load model
    resnet = models.resnet18(pretrained=opt.pretrained)
    model = resnet_adapt.ResNetAdapt(opt, resnet, dataset.class_nb)

    if opt.lr != opt.new_lr:
        model_params = model.lr_params(lr=opt.new_lr)
    else:
        model_params = model.net.parameters()

    optimizer = torch.optim.SGD(model_params, lr=opt.lr,
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

    # Load existing weights, opt.continue_training is epoch to load
    if opt.continue_training:
        if opt.continue_epoch == 0:
            model.net.eval()
            model.load('latest')
        else:
            model.load(opt.continue_epoch)

    train.train_net(dataloader, model, opt,
                    valid_dataloader=valid_dataloader)


if __name__ == "__main__":
    opt = train_options.TrainOptions().parse()
    run_training(opt)
