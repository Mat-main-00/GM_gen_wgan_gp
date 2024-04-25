# Generation of ground motion data by Wasserstein GAN-GP
This repository contains the code and hyperparameters for the paper:

Matsumoto, Y., Yaoyama, T., Lee, S., Hida, T. and Itoi, T. (2023), Fundamental study on probabilistic generative modeling of earthquake ground motion time histories using generative adversarial networks. Jpn Archit Rev, 6: e12392. https://doi.org/10.1002/2475-8876.12392

Please cite this paper if you use the code in this repository as part of a published research project.

## Usage
This repository consists mainly of the following two files.

- model.py
  - Code for model initialization.
- train.py
  - Code for training the model.

To actually perform the training, you need to prepare the training dataset yourself. As an example,
this code import data by specifying the path of data for training in the variable `csv_path` in train.py and defining Dataloader using class `GroundMotionDatasets` in model.py.

## Example Structure of input_files.csv
This section demonstrates an example structure of the input file for loading training data. The input file is defined in the `train.py` file as `csv_path = "../input_files.csv"`. When `input_files.csv` is loaded as a DataFrame using pandas, the result appears as follows:
```
                   file_name            label
0      ../data/example_0.npy       107.130000
1      ../data/example_1.npy       129.400000
2      ../data/example_2.npy        81.543554
3      ../data/example_3.npy       134.679800
4      ../data/example_4.npy       129.400000
...                      ...              ...
21695  ../data/example_21695.npy    32.560853

[21696 rows x 2 columns]
```
The file headers consist of two columns: `file_name` and `label`. Note that the `label` column is included for implementation convenience and is not used in the code of this repository. Hence, the values in the `label` column do not impact the learning process. The `file_name` column contains the paths to the training ground motion data files, with each path corresponding to one set of ground motion data. Therefore, the number of rows equals the number of data in the training dataset.

## Structure of example_*.npy Files
The contents of the example_*.npy files are time-history data for training ground motion. In the code of this repository, these are represented as one-dimensional arrays with a length corresponding to the length of the ground motion data.
