"""
Entry point training and testing multi-scene transformer
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from datasets.C2FCameraPoseDataset import C2FCameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors import get_model
from os.path import join

def test(args, config, model, apply_c2f, num_clusters_position, num_clusters_orientation):
    model.eval()

    # Set the dataset and data loader
    transform = utils.test_transforms.get('baseline')
    if apply_c2f:
        dataset = C2FCameraPoseDataset(args.dataset_path, args.labels_file, transform, False, num_clusters_position, num_clusters_orientation,
                                       args.cluster_predictor_position, args.cluster_predictor_orientation)
    else:
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, False)
    loader_params = {'batch_size': 1,
                     'shuffle': False,
                     'num_workers': config.get('n_workers')}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

    stats = np.zeros((len(dataloader.dataset), 3))

    with torch.no_grad():
        for i, minibatch in enumerate(dataloader, 0):
            for k, v in minibatch.items():
                minibatch[k] = v.to(device)
            minibatch['scene'] = None  # avoid using ground-truth scene during prediction
            minibatch['cluster_id_position'] = None  # avoid using ground-truth cluster during prediction
            minibatch['cluster_id_orientation'] = None  # avoid using ground-truth cluster during prediction


            gt_pose = minibatch.get('pose').to(dtype=torch.float32)

            # Forward pass to predict the pose
            tic = time.time()
            est_pose = model(minibatch).get('pose')
            toc = time.time()

            # Evaluate error
            posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

            # Collect statistics
            stats[i, 0] = posit_err.item()
            stats[i, 1] = orient_err.item()
            stats[i, 2] = (toc - tic) * 1000

            logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                stats[i, 0], stats[i, 1], stats[i, 2]))

    # Record overall statistics
    logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
    logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]),
                                                                    np.nanmedian(stats[:, 1])))
    logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))
    return stats


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")
    arg_parser.add_argument("--cluster_predictor_position", help="path to position k-means predictor")
    arg_parser.add_argument("--cluster_predictor_orientation", help="path to orientation k-means predictor")

    arg_parser.add_argument("--test_dataset_id", default=None, help="test set id for testing on all scenes, options: 7scene OR cambridge")


    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Coarse-to-fine params
    apply_c2f = config.get("c2f")
    num_clusters_position = config.get("nclusters_position")
    num_clusters_orientation = config.get("nclusters_orientation")

    # Create the model
    model = get_model(args.model_name, args.backbone_path, config).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:

        msg = model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id), strict=False)
        logging.info(msg)
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Freeze parts of the model if indicated
        freeze = config.get("freeze")
        freeze_exclude_phrase = config.get("freeze_exclude_phrase")
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                        parameter.requires_grad_(False)


        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)
        nll_loss = torch.nn.NLLLoss()


        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        no_augment = config.get("no_augment")
        if no_augment:
            transform = utils.test_transforms.get('baseline')
        else:
            transform = utils.train_transforms.get('baseline')

        equalize_scenes = config.get("equalize_scenes")
        if apply_c2f:
            dataset = C2FCameraPoseDataset(args.dataset_path, args.labels_file, transform, equalize_scenes,
                                           num_clusters_position, num_clusters_orientation, args.cluster_predictor_position,
                                           args.cluster_predictor_orientation)
        else:
            dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, equalize_scenes)

        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_scene = minibatch.get('scene').to(device)
                position_gt_cluster = minibatch.get('position_cluster_id').to(device)
                orientation_gt_cluster = minibatch.get('orientation_cluster_id').to(device)

                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                if freeze: # For TransPoseNet
                    model.eval()
                    with torch.no_grad():
                        transformers_res = model.forward_transformers(minibatch)
                    model.train()

                # Zero the gradients
                optim.zero_grad()

                # Forward pass to estimate the pose
                if freeze:
                    if apply_c2f:
                        res = model.forward_heads(transformers_res, minibatch)
                    else:
                        res = model.forward_heads(transformers_res)
                else:
                    res = model(minibatch)

                est_pose = res.get('pose')
                est_scene_log_distr = res.get('scene_log_distr')
                est_position_cluster_log_distr = res.get('position_cluster_log_distr')
                est_orientation_cluster_log_distr = res.get('orientation_cluster_log_distr')

                if est_scene_log_distr is not None:
                    # Pose Loss + Scene Loss
                    if apply_c2f:
                        criterion = pose_loss(est_pose, gt_pose) + nll_loss(est_scene_log_distr, gt_scene) \
                                    + nll_loss(est_position_cluster_log_distr, position_gt_cluster) + nll_loss(est_orientation_cluster_log_distr, orientation_gt_cluster)
                    else:
                        criterion = pose_loss(est_pose, gt_pose) + nll_loss(est_scene_log_distr, gt_scene)
                else:
                    # Pose loss
                    criterion = pose_loss(est_pose, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

        # Plot the loss function
        loss_fig_path = checkpoint_prefix + "_loss_fig.png"
        utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

    else: # Test
        if args.test_dataset_id is not None:
            f = open("{}_{}_report.csv".format(args.test_dataset_id,  utils.get_stamp_from_log()), 'w')
            f.write("scene,pos,ori\n")
            if args.test_dataset_id == "7scenes":
                scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
                for scene in scenes:
                    args.cluster_predictor_position = "./datasets/7Scenes/7scenes_all_scenes.csv_scene_{}_position_{}_classes.sav".format(scene, num_clusters_position)
                    args.cluster_predictor_orientation = "./datasets/7Scenes/7scenes_all_scenes.csv_scene_{}_orientation_{}_classes.sav".format(scene, num_clusters_orientation)

                    args.labels_file = "./datasets/7Scenes/abs_7scenes_pose.csv_{}_test.csv".format(scene)
                    stats = test(args, config, model, apply_c2f, num_clusters_position, num_clusters_orientation)
                    f.write("{},{:.3f},{:.3f}\n".format(scene, np.nanmedian(stats[:, 0]),
                                                                                    np.nanmedian(stats[:, 1])))
            elif args.test_dataset_id == "cambridge":

                scenes = ["KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
                for scene in scenes:
                    args.cluster_predictor_position = "./datasets/CambridgeLandmarks/cambridge_four_scenes.csv_scene_{}_position_{}_classes.sav".format(
                        scene, num_clusters_position)
                    args.cluster_predictor_orientation = "./datasets/CambridgeLandmarks/cambridge_four_scenes.csv_scene_{}_orientation_{}_classes.sav".format(
                        scene, num_clusters_orientation)
                    args.labels_file = "./datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{}_test.csv".format(scene)
                    stats = test(args, config, model, apply_c2f, num_clusters_position, num_clusters_orientation)
                    f.write("{},{:.3f},{:.3f}\n".format(scene, np.nanmedian(stats[:, 0]),
                                                                                    np.nanmedian(stats[:, 1])))
            else:
                raise NotImplementedError()
            f.close()
        else:
            _ = test(args, config, model, apply_c2f, num_clusters_position, num_clusters_orientation)







