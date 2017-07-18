import copy
import numpy as np
import time
from tqdm import tqdm

from src.options import error
from src.utils.visualize import Visualize
from src.utils.evaluation import batch_topk_accuracy as topk
from src.utils.evaluation import Metric


def train_net(dataloader, model, criterion, opt,
              optimizer=None, valid_dataloader=None,
              verbose=False):
    top1 = Metric('top1', func=lambda out, pred: topk(out, pred, 1))
    top5 = Metric('top5', func=lambda out, pred: topk(out, pred, 5))
    loss_metric = Metric('loss', compute=False)
    metrics = [top1, top5, loss_metric]
    valid_metrics = copy.deepcopy(metrics)

    # visdom window handles
    sample_win = None
    valid_sample_win = None

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
        for i, (image, target) in enumerate(tqdm(dataloader, desc='iter')):
            metrics, sample_win = data_pass(model, image, target,
                                            opt, metrics=metrics,
                                            dataloader=dataloader,
                                            vis=visualizer,
                                            sample_win=sample_win,
                                            i=i)
        for i, (image, target) in enumerate(tqdm(valid_dataloader, desc='iter')):
            valid_metrics, valid_sample_win = data_pass(model, image,
                                                        target, opt,
                                                        metrics=valid_metrics,
                                                        dataloader=valid_dataloader,
                                                        vis=visualizer,
                                                        sample_win=valid_sample_win,
                                                        i=i, train=False)
        # Compute epoch scores and clear current scores
        for metric in metrics:
            metric.update_epoch()

        for metric in valid_metrics:
            metric.update_epoch()

        last_scores = {metric.name: metric.evolution
                       for metric in metrics}

        # Write scores to log file
        message = visualizer.log_errors(epoch, last_scores)
        if verbose:
            print(message)

        # Display scores in visdom
        for metric in metrics:
            scores = np.array(metric.evolution)
            win = visualizer.plot_errors(np.array(list(range(len(scores)))),
                                         scores, title=metric.name,
                                         win=metric.win)
            metric.win = win

        for metric in valid_metrics:
            scores = np.array(metric.evolution)
            win = visualizer.plot_errors(np.array(list(range(len(scores)))),
                                         scores,
                                         title='valid ' + metric.name,
                                         win=metric.win)
            metric.win = win

        # Save network weights
        if opt.save_latest:
            model.save('latest')
        if epoch % opt.save_freq == 0:
            model.save(epoch)

    if verbose:
        print('Done training')


def data_pass(model, image, target, opt,
              dataloader, i=0, metrics=None, vis=None,
              sample_win=None, train=True):
    image = model.prepare_var(image)
    target = model.prepare_var(target)

    output = model.forward(image)
    loss = model.compute_loss(output, target)

    if train:
        model.step_backward(loss)

    for metric in metrics:
        if metric.compute:
            score = metric.func(output.data, target.data)
            metric.epoch_scores.append(score)
        if metric.name == 'loss':
            metric.epoch_scores.append(loss.data[0])

    # Display an image example in visdom
    if vis is not None and i % opt.display_freq == 0:
        sample_win = vis.plot_sample(image.data,
                                     target.data,
                                     output.data,
                                     dataloader.dataset.classes,
                                     sample_win,
                                     unnormalize=dataloader.dataset.untransform)

    return metrics, sample_win
