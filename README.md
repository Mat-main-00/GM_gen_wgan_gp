# Generation of ground motion data by Wasserstein GAN-GP
This repository consists mainly of the following two files.

- model.py
  - Code for model initialization.
- train.py
  - Code for training the model.

To actually perform the training, you need to prepare the training dataset yourself. As an example,
this code import data by specifying the path of data for training in the variable "csv_path" in train.py and defining Dataloader using class "GroundMotionDatasets" in model.py.
