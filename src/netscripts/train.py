import copy
import numpy as np
from tqdm import tqdm

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
    val_metrics = copy.deepcopy(metrics)

    # visdom window handles
    sample_win = None
    if valid_dataloader is not None:
        val_win = None

    viz = Visualize(opt)
    if opt.use_gpu:
        # Transfert model to GPU
        model = model.cuda()
        criterion = criterion.cuda()

    if opt.train:
        epoch_nb = opt.epochs
    else:
        epoch_nb = 1

    # Initialize conf_mat
    classes = dataloader.dataset.classes
    class_nb = len(classes)
    conf_mat = np.zeros((epoch_nb, class_nb, class_nb))
    val_conf_mat = np.zeros((epoch_nb, class_nb, class_nb))
    conf_win = None
    val_conf_win = None

    for epoch in tqdm(range(epoch_nb), desc='epoch'):
        # Train for one epoch
        metrics, sample_win,\
            conf_mat, conf_win = epoch_pass(dataloader, model, opt,
                                            epoch, metrics, viz,
                                            conf_mat=conf_mat,
                                            conf_win=conf_win,
                                            sample_win=sample_win,
                                            train=True,
                                            verbose=False)
        conf_win = viz.plot_mat(conf_mat[epoch], win=conf_win,
                                title='train conf mat')

        if valid_dataloader is not None:
            # Validation for one epoch
            val_metrics, val_win,\
                val_conf_mat, val_conf_win = epoch_pass(valid_dataloader,
                                                        model,
                                                        opt, epoch,
                                                        conf_mat=val_conf_mat,
                                                        conf_win=val_conf_win,
                                                        metrics=val_metrics,
                                                        viz=viz,
                                                        sample_win=val_win,
                                                        train=False,
                                                        verbose=False)
        val_conf_win = viz.plot_mat(val_conf_mat[epoch], win=val_conf_win,
                                    title='val conf mat')

        # Save network weights
        if opt.save_latest:
            model.save('latest', opt)
        if epoch % opt.save_freq == 0:
            model.save(epoch, opt)

    if verbose:
        print('Done training')


def data_pass(model, image, target, opt,
              dataloader, epoch, i=0, metrics=None,
              viz=None, conf_mat=None,
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

    if conf_mat is not None:
        pred_classes = output.data.max(1)[1].cpu().numpy()
        target_classes = target.data.max(1)[1].cpu().numpy()
        for idx in range(len(pred_classes)):
            conf_mat[epoch, target_classes[idx],
                     pred_classes[idx]] += 1

    # Display an image example in visdom
    if viz is not None and i % opt.display_freq == 0:
        sample_win = viz.plot_sample(image.data,
                                     target.data,
                                     output.data,
                                     dataloader.dataset.classes,
                                     sample_win,
                                     unnormalize=dataloader.dataset.untransform)

    return metrics, sample_win, conf_mat


def epoch_pass(dataloader, model, opt, epoch, metrics, viz,
               sample_win=None, train=True, verbose=False,
               conf_mat=None, conf_win=None):
    for i, (image, target) in enumerate(tqdm(dataloader, desc='iter')):
        metrics, sample_win,\
            conf_mat = data_pass(model, image, target,
                                 opt, epoch=epoch,
                                 metrics=metrics,
                                 dataloader=dataloader,
                                 viz=viz,
                                 conf_mat=conf_mat,
                                 sample_win=sample_win,
                                 i=i, train=train)

    # Display confusion matrix
    epoch_conf_mat = conf_mat[epoch]
    conf_win = viz.plot_mat(epoch_conf_mat, conf_win)

    # Compute epoch scores and clear current scores
    for metric in metrics:
        metric.update_epoch()

    last_scores = {metric.name: metric.evolution[-1]
                   for metric in metrics}

    # Display scores in visdom
    for metric in metrics:
        plt_title = 'valid ' + metric.name if train is False else metric.name
        scores = np.array(metric.evolution)
        win = viz.plot_errors(np.array(list(range(len(scores)))),
                              scores, title=plt_title,
                              win=metric.win)
        metric.win = win

    # Write scores to log file
    valid = True if train is False else False
    message = viz.log_errors(epoch, last_scores, valid=valid)

    # Sanity check, top1 score should be the same as accuracy from conf_mat
    assert last_scores['top1'] == epoch_conf_mat.trace() / epoch_conf_mat.sum()

    if verbose:
        print(message)

    return metrics, sample_win, conf_mat, conf_win
