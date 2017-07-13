import numpy as np
import torch
from tqdm import tqdm


def train_net(dataloader, model, optimizer, criterion,
              epochs=10, verbose=True, use_gpu=True):
    loss_evolution = []
    if use_gpu:
        # Transfert model to GPU
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in tqdm(range(epochs), desc='epoch'):
        losses = []
        for i, (image, target) in enumerate(tqdm(dataloader, position=1)):
            # Cast from double to float
            target = target.float()
            # Transfer to GPU
            if use_gpu:
                target = target.cuda()
                image = image.cuda()

            # Create pytorch Varibles
            input_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)

            # Forward pass
            output = model(input_var)
            loss = criterion(output, target_var)
            losses.append(loss.data[0])

            # compute gradient and do descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = np.mean(losses)
        loss_evolution.append(mean_loss)
        print('Loss : {0}'.format(mean_loss))

    if verbose:
        print('Done training')
    return loss_evolution
