import copy
import numpy as np
import pickle
import os
from tqdm import tqdm

from src.utils.visualize import Visualize
from src.utils.evaluation import batch_topk_accuracy as topk
from src.utils.evaluation import Metric
from src.netscripts.test import test


def train_net(dataloader,
              model,
              opt,
              valid_dataloader=None,
              verbose=False,
              visualize=True,
              test_aggreg=True):
    """
    Args:
        visualize(bool): whether to display in visdom
    """
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
        model.net = model.net.cuda()
        model.criterion = model.criterion.cuda()

        last_epoch = opt.epochs

    # Initialize conf_mat
    classes = dataloader.dataset.classes
    class_nb = len(classes)
    conf_win = None
    val_conf_win = None

    valid_mean_scores = []
    valid_mean_win = None

    if opt.continue_training:
        start_epoch = opt.continue_epoch
    else:
        start_epoch = 0

    # Conf mat are overwritten when training is continued
    conf_mat = np.zeros((last_epoch, class_nb, class_nb))
    val_conf_mat = np.zeros((last_epoch, class_nb, class_nb))

    for epoch in tqdm(range(start_epoch, last_epoch), desc='epoch'):
        # Train for one epoch
        metrics, sample_win,\
            conf_mat, conf_win = epoch_pass(dataloader, model, opt,
                                            epoch, metrics, viz,
                                            conf_mat=conf_mat,
                                            conf_win=conf_win,
                                            sample_win=sample_win,
                                            train=True,
                                            verbose=False,
                                            visualize=visualize)
        if visualize:
            conf_win = viz.plot_mat(
                conf_mat[epoch], win=conf_win, title='train conf mat')

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
                                                        verbose=False,
                                                        visualize=visualize)
        # Display valid conf mat
        if visualize:
            val_conf_win = viz.plot_mat(
                val_conf_mat[epoch], win=val_conf_win, title='val conf mat')

        # Save network weights
        if opt.save_latest:
            model.save(epoch, opt, latest=True)
        if epoch % opt.save_freq == 0:
            model.save(epoch, opt)

        # Save confusion matrixes
        save_dir = os.path.join(opt.checkpoint_dir, opt.exp_id)
        train_conf_path = os.path.join(save_dir, 'train_conf_mat.pickle')
        val_conf_path = os.path.join(save_dir, 'val_conf_mat.pickle')
        with open(train_conf_path, 'wb') as train_conf_file:
            pickle.dump(conf_mat, train_conf_file)
        with open(val_conf_path, 'wb') as val_conf_file:
            pickle.dump(conf_mat, val_conf_file)

        # Test with aggregation
        if test_aggreg:
            valid_mean_score = test(
                valid_dataloader.dataset, model, frame_nb=10, opt=opt)
            valid_mean_scores.append(valid_mean_score)

            # Display and save validations info
            viz.log_errors(
                epoch=epoch,
                errors={'val_aggr_err': valid_mean_score},
                log_path=viz.valid_aggreg_log_path)
            if visualize:
                valid_mean_win = viz.plot_errors(
                    np.array(list(range(epoch + 1))),
                    np.array(valid_mean_scores),
                    title='average aggreg acc',
                    win=valid_mean_win)

    if verbose:
        print('Done training')


def data_pass(model,
              image,
              target,
              opt,
              dataloader,
              epoch,
              i=0,
              metrics=None,
              viz=None,
              conf_mat=None,
              sample_win=None,
              train=True,
              visualize=True):
    """
    Args:
        visualize(bool): whether to display in visdom
    """
    image = model.prepare_var(image)
    target = model.prepare_var(target)

    output = model.net.forward(image)
    loss = model.compute_loss(output, target)

    if train:
        model.step_backward(loss)

    for metric in metrics:
        if metric.compute:
            score = metric.func(output.data, target.data)
            metric.epoch_scores.append((score.sum(), len(score)))
        if metric.name == 'loss':
            metric.epoch_scores.append((loss.data[0] * len(score), len(score)))

    if conf_mat is not None:
        pred_classes = output.data.max(1)[1].cpu().numpy()
        target_classes = target.data.max(1)[1].cpu().numpy()
        for idx in range(len(pred_classes)):
            conf_mat[epoch, target_classes[idx], pred_classes[idx]] += 1

    # Display an image example in visdom
    if visualize:
        if viz is not None and i % opt.display_freq == 0:
            sample_win = viz.plot_sample(
                image.data,
                target.data,
                output.data,
                dataloader.dataset.classes,
                sample_win,
                unnormalize=dataloader.dataset.untransform)

    return metrics, sample_win, conf_mat


def epoch_pass(dataloader,
               model,
               opt,
               epoch,
               metrics,
               viz,
               sample_win=None,
               train=True,
               verbose=False,
               conf_mat=None,
               conf_win=None,
               visualize=True):
    if train:
        model.net.train()
    else:
        model.net.eval()
    for i, (image, target) in enumerate(tqdm(dataloader, desc='iter')):
        metrics, sample_win,\
            conf_mat = data_pass(model, image, target,
                                 opt, epoch=epoch,
                                 metrics=metrics,
                                 dataloader=dataloader,
                                 viz=viz,
                                 conf_mat=conf_mat,
                                 sample_win=sample_win,
                                 i=i, train=train,
                                 visualize=visualize)

    # Display confusion matrix
    epoch_conf_mat = conf_mat[epoch]
    if visualize:
        conf_win = viz.plot_mat(epoch_conf_mat, conf_win)

    # Compute epoch scores and clear current scores
    for metric in metrics:
        metric.update_epoch()

    last_scores = {metric.name: metric.evolution[-1] for metric in metrics}

    # Display scores in visdom
    for metric in metrics:
        plt_title = 'valid ' + metric.name if train is False else metric.name
        scores = np.array(metric.evolution)
        if visualize:
            win = viz.plot_errors(
                np.array(list(range(len(scores)))),
                scores,
                title=plt_title,
                win=metric.win)
            metric.win = win

    # Write scores to log file
    valid = True if train is False else False
    message = viz.log_errors(epoch, last_scores, valid=valid)

    # Sanity check, top1 score should be the same as accuracy from conf_mat
    # while accounting for last batch discrepancy

    if last_scores['top1'] != epoch_conf_mat.trace() / epoch_conf_mat.sum():
        import pdb
        pdb.set_trace()
    assert last_scores['top1'] == epoch_conf_mat.trace() / epoch_conf_mat.sum(
    ), '{} is not {}'.format(last_scores['top1'],
                             epoch_conf_mat.trace() / epoch_conf_mat.sum())

    if verbose:
        print(message)

    viz.save()

    return metrics, sample_win, conf_mat, conf_win
