# Deep Learning Exam - Cross-Stitch Networks

This project implements **Cross-Stitch Networks** ([paper link](https://arxiv.org/abs/1604.03539)) for the following tasks: **Object Detection** and **Attribute Predictions**. The aim is to compare the single-task and multi-task versions and replicate the paper's results for these tasks.

## Project Structure

- **`config/`**: Contains the `.yaml` file with the experiment settings.
- **`data/`**: Contains the annotated dataset.
- **`experiments/`**: Contains the results of each experiment performed (initially empty).
- **`script/`**: Contains the `.sh` files to run training experiments.
- **`selective_search/`**: Contains the box proposals for each image for each split (train/val) (initially empty).
- **`src/`**: Source code for single and multi-task learning.
- **`EDA_VOC2008_Attribute.ipynb`**: Exploratory data analysis of the dataset used.

## Datasets

The dataset used is **PASCAL VOC 2008 Attributes** available at the following [Link](https://vision.cs.uiuc.edu/attributes/).

You need to download "aPascal images" and "Annotations".

**NOTE 1**: The training set corresponds to the VOC2008 train set, and the test set corresponds to the VOC2008 validation set.  
**NOTE 2**: If the download link doesn't work, you can use [Wayback Machine](https://web.archive.org/) to retrieve a previous version of the website (e.g., in 2022, the website worked).

After downloading, create a `data/VOC2008_attribute/` folder with the following structure:
- **`ann/`**: Contains `ann_train.txt` and `ann_val.txt`, which are the annotations of the dataset.
- **`train/`**: Images of the train split.
- **`val/`**: Images of the val/test split.
- **`attribute_names.txt`**: List of attributes for attribute prediction.
- **`class_names.txt`**: List of classes for object detection.

## Installation

### Create Environment

You need to create a conda environment with the following packages:

```bash
- torch
- torchmetrics
- numpy
- yaml
- wandb
- tqdm
- PIL
- pandas
- matplotlib
- sklearn
- netron
```

## Selective-seaarch Boxes

You have to install selective search package:

```bash 
pip install selective-search
```

and after run **`src/object_detection/selective_search.py`**  in order to create the file containing the proposals boxes.
You will obtain a folder "**`selectivesearch/`**" containing 2 files .pt

## Training

The project is split into 3 parts:
- **Single task - Object Detection**
- **Single task - Attribute Prediction**
- **Multi-task - Cross-Stitch Network with the previous 2 tasks**

Once you have selected the task(s) to solve, you can run the corresponding `.sh` file in the `script/` folder after editing the `.yaml` file in the `config/` folder.

After training, a datetime folder will be created in the `experiments/` folder containing the output of the experiment and the model checkpoint (including the best model checkpoint).

## Evaluation

After training your model, you can evaluate it through `eval.ipynb`. Once you edit the first code block with your model path, you can run all the cells, and the last cells will show the quantitative results.

## Optional - Netron

For each task, you can view the model architecture by running all cells of the `netron.ipynb`. This generates a `.onnx` file (follow [Netron GitHub](https://github.com/lutzroeder/netron) for details).

## Optional - Qualitative Results

For each task, you can view some qualitative results by running all cells of the `qualitative_results.ipynb`. The images will be displayed directly in the notebook.