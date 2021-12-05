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

    def __init__(self, dataset_path, labels_file, data_transform=None, equalize_scenes=False, num_position_clusters=4, num_orientation_clusters=4,
                 kmeans_position_file=None, kmeans_orientation_file=None):
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
        self.orientation_centroids = {}
        self.orientation_cluster_ids = np.zeros(self.dataset_size)
        for i in range(self.num_scenes):
            locs = np.array(self.scenes_ids) == i
            if kmeans_position_file is None:
                scene_positions = self.poses[locs, :3]
                kmeans_position = KMeans(n_clusters=num_position_clusters, random_state=random_state).fit(scene_positions)
                filename = labels_file + '_scene_{}_position_{}_classes.sav'.format(self.scenes[locs][0], num_position_clusters)
                print(filename)
                joblib.dump(kmeans_position, filename)

                scene_orientations = self.poses[locs, 3:]
                kmeans_orientation = KMeans(n_clusters=num_orientation_clusters, random_state=random_state).fit(
                    scene_orientations)
                filename = labels_file + '_scene_{}_orientation_{}_classes.sav'.format(self.scenes[locs][0],
                                                                                    num_orientation_clusters)
                print(filename)
                joblib.dump(kmeans_orientation, filename)



            else:
                kmeans_position = joblib.load(kmeans_position_file)
                kmeans_orientation = joblib.load(kmeans_orientation_file)


            self.position_centroids[i] = kmeans_position.cluster_centers_.astype(np.float32)
            self.position_cluster_ids[locs] = kmeans_position.predict(self.poses[locs, :3]).astype(np.int)
            self.orientation_centroids[i] = kmeans_orientation.cluster_centers_.astype(np.float32)
            self.orientation_cluster_ids[locs] = kmeans_orientation.predict(self.poses[locs, 3:]).astype(np.int)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        if idx >= len(self.poses): # sample from an under-repsented scene
            sampled_scene_idx = np.random.choice(range(self.num_scenes), p=self.scene_prob_selection)
            idx = np.random.choice(self.scenes_sample_indices[sampled_scene_idx])

        img = imread(self.img_paths[idx])
        pose = self.poses[idx]
        position_cluster_id = int(self.position_cluster_ids[idx])
        orientation_cluster_id = int(self.orientation_cluster_ids[idx])
        scene = self.scenes_ids[idx]
        position_centroids = self.position_centroids[scene]
        orientation_centroids = self.orientation_centroids[scene]

        if self.transform:
            img = self.transform(img)

        sample = {'img': img, 'pose': pose, 'scene': scene,
                  'position_centroids':position_centroids, 'position_cluster_id': position_cluster_id,
                  'orientation_centroids': orientation_centroids, 'orientation_cluster_id': orientation_cluster_id
                  }
        return sample


