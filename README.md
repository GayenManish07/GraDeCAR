# GraDeCAR:  Gradual Denoising by Contrastive Agreement-based Relabeling

This is the official repository of GraDeCar method implementation.


## Method Overview

1. Confident samples are selected using Cleanlab and a pretrained EfficientNet.
2. Two models are trained in parallel: a contrastive model and a CNN classifier.
3. Agreement-based relabeling is used for non-confident samples.
4. Relabeling is followed by annealing training epochs across multiple rounds.
5. Final predictions are obtained by averaging softmax outputs from both models.

For deeper insight please refer to our paper.

## Running the code

### Input data

The code expects the data to be stored in the following format:
```text
project_root/
├── train.csv
├── val.csv
├── test.csv
└── images/
    ├── image1.png
    ├── image2.png
    ├── ...
```
Where each image has a unique file name and the csv contains two columns comprising of image file name and the class number against it.

The loader.py facilitates the use of this specific data format in our method. It can be used with any image data as long as it is stored in the aforementioned structure, using the loader file.
The loader file also injects label noise to the training data according to the run configuration.

### Output

On inference the code outputs a csv file containing the predicted class against the instance index.

### Run command
```bash
python main.py --noise_type 'structured' --noise_rate 0.3
```
```bash
python main.py --noise_type 'symmetric' --noise_rate 0.3
```

For changing more hyperparameters refer to the arguments accepted by the main file.

## Data

The code is designed to work on the APTOS 2019 dataset which can be found [here](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
