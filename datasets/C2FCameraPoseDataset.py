from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
from datasets.CameraPoseDataset import CameraPoseDataset
from sklearn.cluster import KMeans
import joblib


class C2FCameraPoseDataset(CameraPoseDataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, dataset_path, labels_file, data_transform=None, equalize_scenes=False, num_position_clusters=4, kmeans_position_file=None):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param data_transform: (Transform object) a torchvision transform object
        :return: an instance of the class
        """
        super(C2FCameraPoseDataset, self).__init__(dataset_path, labels_file, data_transform, equalize_scenes)
        random_state = 170

        # Generate clusters for each scene
        self.position_centroids = {}
        self.position_cluster_ids = np.zeros(self.dataset_size)
        for i in range(self.num_scenes):
            locs = np.array(self.scenes_ids) == i
            if kmeans_position_file is None:
                scene_positions = self.poses[locs, :3]

                kmeans_position = KMeans(n_clusters=num_position_clusters, random_state=random_state).fit(scene_positions)

                filename = labels_file + 'scene_{}_position_{}_classes.sav'.format(self.scenes[i], num_position_clusters)
                joblib.dump(kmeans_position, filename)

            else:
                kmeans_position = joblib.load(kmeans_position_file)

            self.position_centroids[i] = kmeans_position.cluster_centers_
            self.position_cluster_ids[locs] = kmeans_position.predict(self.poses[locs, :3]).astype(np.int)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        if idx >= len(self.poses): # sample from an under-repsented scene
            sampled_scene_idx = np.random.choice(range(self.num_scenes), p=self.scene_prob_selection)
            idx = np.random.choice(self.scenes_sample_indices[sampled_scene_idx])

        img = imread(self.img_paths[idx])
        pose = self.poses[idx]
        cluster_id = self.position_cluster_ids[idx]
        scene = self.scenes_ids[idx]
        centroids = self.position_centroids[scene]

        if self.transform:
            img = self.transform(img)

        sample = {'img': img, 'pose': pose, 'scene': scene, 'centroids':centroids, 'cluster_id': cluster_id}
        return sample


