## A Coarse-to-Fine Residual Prediction for Multi-Scene Pose Regression with Transformers 
This repository extends the ICCV21 paper [Learning Multi-Scene Absolute Pose Regression with Transformers](https://arxiv.org/abs/2103.11468) with coarse-to-fine position residual regression.


### Usage

The entry point for training and testing is the main.py script in the root directory

  For detailed explanation of the options run:
  ```
  python main.py -h
  ```
  For example, in order to train our model on the 7Scenes dataset run: 
  ```
python main.py ems-transposenet train models/backbones/efficient-net-b0.pth /path/to/7scenes-datasets ./datasets/7Scenes/7scenes_all_scenes.csv 7Scenes_config.json
  ```
  Your checkpoints (.pth file saved based on the number you specify in the configuration file) and log file
  will be saved under an 'out' folder.
  
  To run on cambridge, you will need to change the configuration file to ```CambridgeLandmarks_config.json``` for initial training and ```CambridgeLandmarks_finetune_config.json``` for fine-tuning (see details in our paper). 
  
  In order to test your model, for example on the fire scene from the 7Scenes dataset:
  ```
  python main.py ems-transposenet test /./models/backbones/efficient-net-b0.pth /path/to/7scenes-datasets ./datasets/7Scenes/abs_7scenes_pose.csv_fire_test.csv 7Scenes_config.json --checkpoint_path <path to your checkpoint .pth>
  ```
 ### Citation 
 If you find this repository useful, please consider giving a star and citation:
```
@article{Shavit21,
  title={Learning Multi-Scene Absolute Pose Regression with Transformers},
  author={Shavit, Yoli and Ferens, Ron and Keller, Yosi},
  journal={arXiv preprint arXiv:2103.11468},
  year={2021}
}
  
  
  
  
