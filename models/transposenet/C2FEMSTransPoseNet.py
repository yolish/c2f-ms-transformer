"""
The Coarse-to-Fine Efficient Multi-Scene TransPoseNet model
"""

import torch
from .MSTransPoseNet import PoseRegressor
from .EMSTransPoseNet import EMSTransPoseNet


def add_residuals(cls_log_distr, centroids, redisuals, gt_indices=None):
    batch_size = cls_log_distr.shape[0]
    _, max_indices = cls_log_distr.max(dim=1)
    # Take the global latents by zeroing other scene's predictions and summing up
    w = centroids * 0
    if gt_indices is not None:
        max_indices = gt_indices
    w[range(batch_size), max_indices] = 1
    selected_centroids = torch.sum(w * centroids, dim=1)
    return selected_centroids + redisuals


def select_centroids(cls_log_distr, centroids):
    batch_size = cls_log_distr.shape[0]
    _, max_indices = cls_log_distr.max(dim=1)
    # Take the global latents by zeroing other scene's predictions and summing up
    w = centroids * 0
    w[range(batch_size), max_indices] = 1
    selected_centroids = torch.sum(w * centroids, dim=1)
    return selected_centroids


class C2FEMSTransPoseNet(EMSTransPoseNet):

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__(config, pretrained_path)

        decoder_dim = self.transformer_t.d_model
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4)
        self.t_cluster_embed = torch.nn.Linear(decoder_dim, config.get("nclusters_position"))
        self.rot_cluster_embed = torch.nn.Linear(decoder_dim, config.get("nclusters_orientation"))


    def forward_heads(self, transformers_res, data):
        """
        Forward pass of the MLP heads
        The forward pass execpts a dictionary with two keys-values:
        global_desc_t: latent representation from the orientation encoder
        global_dec_rot: latent representation from the orientation encoder
        scene_log_distr: the log softmax over the scenes
        max_indices: the index of the max value in the scene distribution
        returns: dictionary with key-value 'pose'--expected pose (NX7) and scene_log_distr
        """
        global_desc_t = transformers_res.get('global_desc_t')
        global_desc_rot = transformers_res.get('global_desc_rot')

        position_centroids = data["position_centroids"]

        gt_position_cluster_ids = data.get("position_cluster_id") # None at Test time

        # predict position cluster and residual
        t_cluster_log_distr = self.log_softmax(
            self.t_cluster_embed(global_desc_t))
        t_residuals = self.regressor_head_t(global_desc_t)

        # Regress the pose
        x_t = add_residuals(t_cluster_log_distr, position_centroids,
                            t_residuals, gt_indices=gt_position_cluster_ids)

        orientation_centroids = data["orientation_centroids"]

        gt_orientation_cluster_ids = data.get("orientation_cluster_id")  # None at Test time

        # predict orientation cluster and residual
        rot_cluster_log_distr = self.log_softmax(
            self.rot_cluster_embed(global_desc_rot))
        rot_residuals = self.regressor_head_rot(global_desc_rot)

        # Regress the pose
        x_rot = add_residuals(rot_cluster_log_distr, orientation_centroids,
                            rot_residuals, gt_indices=gt_orientation_cluster_ids)

        expected_pose = torch.cat((x_t, x_rot), dim=1)
        return {'pose':expected_pose, 'scene_log_distr':transformers_res.get('scene_log_distr'),
                "position_cluster_log_distr": t_cluster_log_distr, "orientation_cluster_log_distr": rot_cluster_log_distr}


    def forward(self, data):
        """ The forward pass expects a dictionary with the following keys-values
         'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
         'scene_indices': ground truth scene indices for each image (can be None)

         returns a dictionary with the following keys-values;
        'pose': expected pose (NX7)
        'log_scene_distr': (log) probability distribution over scenes
        """
        transformers_res = self.forward_transformers(data)
        # Regress the pose from the image descriptors

        heads_res = self.forward_heads(transformers_res, data)

        return heads_res