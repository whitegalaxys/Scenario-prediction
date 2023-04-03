## Dataset
Download the [Waymo Open Motion Dataset](https://waymo.com/open/download/) v1.1; only the files in ```uncompressed/scenario/training_20s``` are needed. Place the downloaded files into training and testing folders separately.

## Installation
### Install dependency
```bash
sudo apt-get install libsuitesparse-dev
```

### Create conda env
```bash
conda env create -f environment.yml
conda activate DIPP
```

### Install Theseus
Install the [Theseus library](https://github.com/facebookresearch/theseus), follow the guidelines.

## Usage
### Data Processing
Run ```data_process.py``` to process the raw data for training. This will convert the original data format into a set of ```.npz``` files, each containing the data of a scene with the AV and surrounding agents. You need to specify the file path to the original data ```--load_path``` and the path to save the processed data ```--save_path``` . You can optionally set ```--use_multiprocessing``` to speed up the processing. 
```shell
bash data_process.sh
```

### Training
Run ```bash train.sh``` to learn the predictor. You need to specify the file paths to training data ```--train_set``` and validation data ```--valid_set```. Leave other arguments vacant to use the default setting.
```shell
bash train.sh
```

### Validation
Run ```bash valid.sh``` to do validation for the predictor.
```shell
bash valid.sh
```

### visualization
Run ```bash visualization.sh```to generate the demo video of the scene.
```shell
bash visualization.sh
```
