import numpy as np
import time
from tqdm import tqdm

from src.options import error
from src.utils.visualize import Visualize
from src.utils.evaluation import batch_topk_accuracy as topk


def train_net(dataloader, model, criterion, opt,
              optimizer=None, verbose=False):
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
    sample_win = None

    visualizer = Visualize(opt)
    if opt.use_gpu:
        # Transfert model to GPU
        model = model.cuda()
        criterion = criterion.cuda()

    if opt.train:
        epoch_nb = opt.epochs
    else:
        epoch_nb = 1

    for epoch in tqdm(range(epoch_nb), desc='epoch'):
        losses = []

        # Initialize epoch losses
        for metric in metrics.values():
            metric['epoch_scores'] = []

        for i, (image, target) in enumerate(tqdm(dataloader, desc='iter')):
            image = model.prepare_var(image)
            target = model.prepare_var(target)

            output = model.forward(image)
            loss = model.compute_loss(output, target)

            if opt.train:
                model.step_backward(loss)

            # Compute batch scores
            losses.append(loss.data[0])

            for metric in metrics.values():
                score = metric['func'](output.data, target.data)
                metric['epoch_scores'].append(score)
            # Display an image example in visdom

            if i % opt.display_freq == 0:
                sample_win = visualizer.plot_sample(image.data,
                                                    target.data,
                                                    output.data,
                                                    dataloader.dataset.classes,
                                                    sample_win,
                                                    unnormalize=dataloader.dataset.untransform)

        # Compute epoch mean scores
        mean_loss = np.mean(losses)
        loss_evolution.append(mean_loss)
        for metric in metrics.values():
            metric['evolution'].append(np.mean(metric['epoch_scores']))

        last_scores = {key: metric['evolution'][-1]
                       for key, metric in metrics.items()}
        last_scores['loss'] = mean_loss

        # Write scores to log file
        message = visualizer.log_errors(epoch, last_scores)
        if verbose:
            print(message)

        # Display loss in visdom
        error_win = visualizer.plot_errors(np.array(list(range(epoch + 1))),
                                           np.array(loss_evolution),
                                           title='loss', win=error_win)

        # Display scores in visdom
        for metric_name, metric in metrics.items():
            scores = np.array(metric['evolution'])
            win = visualizer.plot_errors(np.array(list(range(len(scores)))),
                                         scores, title=metric_name,
                                         win=metric['win'])
            metric['win'] = win

        # Save network weights
        if opt.save_latest:
            model.save('latest')
        if epoch % opt.save_freq == 0:
            model.save(epoch)

    if verbose:
        print('Done training')
    return loss_evolution
