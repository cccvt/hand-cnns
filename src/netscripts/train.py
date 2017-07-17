import numpy as np
import time
import torch
from tqdm import tqdm

from src.options import error
from src.utils.visualize import Visualize
from src.utils.evaluation import batch_topk_accuracy as topk


def train_net(dataloader, model, optimizer, criterion,
              opt, verbose=False):
    loss_evolution = []
    metrics = {'top1': {'win': None,
                        'func': lambda out, pred: topk(out, pred, 1),
                        'epoch_scores': [],
                        'evolution': []},
               'top5': {'win': None,
                        'func': lambda out, pred: topk(out, pred, 5),
                        'epoch_scores': [],
                        'evolution': []}}

    # visdom window handles
    error_win = None

    visualizer = Visualize(opt)
    if opt.use_gpu:
        # Transfert model to GPU
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in tqdm(range(opt.epochs), desc='epoch'):
        losses = []

        debug_counter = 0
        # Initialize epoch losses
        for metric in metrics.values():
            metric['epoch_scores'] = []

        for i, (image, target) in enumerate(tqdm(dataloader, desc='iter')):
            # Cast from double to float
            target = target.float()

            # Transfer to GPU
            if opt.use_gpu:
                target = target.cuda()
                image = image.cuda()

            # Create pytorch Varibles
            input_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)

            # Forward pass
            output = model(input_var)

            # Compute scores
            if opt.criterion == 'MSE':
                loss = criterion(output, target_var)
            elif opt.criterion == 'CE':
                # CE expects index of class as ground truth input
                target_vals, target_idxs = target_var.max(1)
                loss = criterion(output, target_idxs.view(-1))
            else:
                raise error.ArgumentError(
                    '{0} is not among known error\
                    functions'.format(opt.criterion))

            losses.append(loss.data[0])
            for metric in metrics.values():
                score = metric['func'](output, target)
                metric['epoch_scores'].append(score)

            # compute gradient and do descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            debug_counter = debug_counter + 1
            # if debug_counter > 100:
            #     break

        # Log metrics
        mean_loss = np.mean(losses)
        loss_evolution.append(mean_loss)
        for metric in metrics.values():
            metric['evolution'].append(np.mean(metric['epoch_scores']))

        last_scores = {key: metric['evolution'][-1]
                       for key, metric in metrics.items()}
        last_scores['loss'] = mean_loss
        message = visualizer.log_errors(epoch, last_scores)
        if verbose:
            print(message)

        error_win = visualizer.plot_errors(np.array(list(range(epoch + 1))),
                                           np.array(loss_evolution),
                                           title='loss', win=error_win)

        # Update visdom displays
        for metric_name, metric in metrics.items():
            scores = np.array(metric['evolution'])
            win = visualizer.plot_errors(np.array(list(range(len(scores)))),
                                         scores, title=metric_name,
                                         win=metric['win'])
            metric['win'] = win
        if opt.save_latest:
            model.save('latest')
        if epoch % opt.save_freq == 0:
            model.save(epoch)

    if verbose:
        print('Done training')
    return loss_evolution
