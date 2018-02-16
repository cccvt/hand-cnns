import copy
import numpy as np
import pickle
import os
from tqdm import tqdm

from src.utils.visualize import Visualize
from src.utils.evaluation import batch_topk_accuracy as topk
from src.utils.evaluation import Metric, Metrics
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

    train_metrics = Metrics('train')
    val_metrics = Metrics('val')

    viz = Visualize(opt)
    if opt.use_gpu:
        # Transfert model to GPU
        model.net = model.net.cuda()
        model.criterion = model.criterion.cuda()

        last_epoch = opt.epochs

    # Initialize conf_mat
    classes = dataloader.dataset.classes
    class_nb = len(classes)

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
        conf_mat = epoch_pass(
            dataloader,
            model,
            opt,
            epoch,
            metrics=train_metrics,
            viz=viz,
            conf_mat=conf_mat,
            train=True,
            verbose=False,
            visualize=visualize)
        if visualize:
            viz.plot_mat(
                'train_confmat', conf_mat[epoch], title='train conf mat')

        if valid_dataloader is not None:
            # Validation for one epoch
            val_conf_mat = epoch_pass(
                valid_dataloader,
                model,
                opt,
                epoch,
                conf_mat=val_conf_mat,
                metrics=val_metrics,
                viz=viz,
                train=False,
                verbose=False,
                visualize=visualize)

        # Update learning rate scheduler according to loss
        if model.lr_scheduler is not None:
            # Retrieve latest loss for lr scheduler
            loss_metric = val_metrics.metrics['loss']
            loss = loss_metric.evolution[-1]
            lr = model.scheduler_step(loss)
            viz.log_errors(
                epoch=epoch,
                errors={'lr': lr,
                        'loss': loss},
                log_path=viz.lr_history_path)

        # Display valid conf mat
        if visualize:
            viz.plot_mat(
                'val_confmat', val_conf_mat[epoch], title='val conf mat')

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
            pickle.dump(val_conf_mat, val_conf_file)

        # Test with aggregation
        if test_aggreg:
            valid_mean_score = test(valid_dataloader.dataset, model, opt=opt)
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
              train=True,
              visualize=True):
    """
    Args:
    visualize(bool): whether to display in visdom
    """
    if train:
        volatile = False
    else:
        volatile = True

    image = model.prepare_var(image, volatile=volatile)
    target = model.prepare_var(target, volatile=volatile)

    output = model.net.forward(image)
    loss = model.compute_loss(output, target)

    if train:
        model.step_backward(loss, opt.multi_weights)

    top1_score = topk(output.data, target.data, 1)
    top5_score = topk(output.data, target.data, 5)
    metrics.add_metric_score('top1', (top1_score.sum(), len(top1_score)))
    metrics.add_metric_score('top5', (top5_score.sum(), len(top1_score)))
    metrics.add_metric_score('loss', (loss.data[0] * len(top1_score),
                                      len(top1_score)))
    if isinstance(target, (tuple, list)):
        output = output[0]
        target = target[0]
        top1_score = topk(output.data, target.data, 1)
        top5_score = topk(output.data, target.data, 5)
        metrics.add_metric_score('top1', (top1_score.sum(), len(top1_score)))
        metrics.add_metric_score('top5', (top5_score, len(top1_score)))
        # elif isinstance(target, dict):
        #   for idx, (target_name, target_value) in enumerate(target.items()):
        #         score = metric.func(output[idx].data, target_value)
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        metrics.add_metric_score('loss', (loss.data[0] * len(top1_score),
                                          len(top1_score)))

    if conf_mat is not None:
        pred_classes = output.data.max(1)[1].cpu().numpy()
        target_classes = target.data.max(1)[1].cpu().numpy()
        for idx in range(len(pred_classes)):
            conf_mat[epoch, target_classes[idx], pred_classes[idx]] += 1

    # Display an image example in visdom
    if visualize:
        if viz is not None and i % opt.display_freq == 0:
            prefix = 'train_' if train else 'val_'
            sample_name = prefix + 'sample'
            viz.plot_sample(sample_name, image.data, target.data, output.data,
                            dataloader.dataset.classes)

    return conf_mat


def epoch_pass(dataloader,
               model,
               opt,
               epoch,
               metrics,
               viz,
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
        conf_mat = data_pass(
            model,
            image,
            target,
            opt,
            epoch=epoch,
            metrics=metrics,
            dataloader=dataloader,
            viz=viz,
            conf_mat=conf_mat,
            i=i,
            train=train,
            visualize=visualize)

    # Display confusion matrix
    epoch_conf_mat = conf_mat[epoch]
    if visualize:
        viz.plot_mat('train_confmat', epoch_conf_mat, title='train conf mat')

    # Compute epoch scores and clear current scores
    for metric in metrics.metrics.values():
        metric.update_epoch()

    last_scores = {
        metric.name: metric.evolution[-1]
        for metric in metrics.metrics.values()
    }

    # Display scores in visdom

    for metric_name, metric in metrics.metrics.items():
        plt_title = '{}_{}'.format(metrics.name, metric.name)
        scores = np.array(metric.evolution)
        if visualize:
            viz.plot_errors(
                plt_title,
                np.array(list(range(len(scores)))),
                scores,
                title=plt_title)

    # Write scores to log file
    valid = True if train is False else False
    message = viz.log_errors(epoch, last_scores, valid=valid)

    # Sanity check, top1 score should be the same as accuracy from conf_mat
    # while accounting for last batch discrepancy

    if last_scores['top1'] != epoch_conf_mat.trace() / epoch_conf_mat.sum():
        import pdb
        pdb.set_trace()

    if verbose:
        print(message)

    viz.save()

    return conf_mat
