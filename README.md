# How to use

Download the needed datasets and put them in the data folder

## Dependencies

This project relies on

- pytorch
- opencv
- visdom

and other packages !

## Datasets

All datasets should be in the data folder.

For instance, for UCI-EGO, it should go into  data/UCI-EGO

For the datasets, the folders should be arranged as following

data

├── [UCI-EGO](https://github.com/hassony2/inria-research-wiki/wiki/uci-ego-dataset)

├── [GTEA](https://github.com/hassony2/inria-research-wiki/wiki/gtea-dataset)

├── [GTEAGazePlus](https://github.com/hassony2/inria-research-wiki/wiki/gtea-gaze-plus-dataset)

├── [gun](http://www.gregrogez.net/research/egovision4health/gun-71/)


Click on the names of the datasets for more info


## Use scripts

### Existing scripts

For now two scripts are provided:

- train.py performs classification from RGB images using a pretrained ResNet with just the last layer tweaked to have the right number of outputs
- c3d\_train.py takes consecutive RGB frames from video to perform action recognition, see [original project page](http://vlg.cs.dartmouth.edu/c3d/)

###  Get help

To get information about the various options, use

`python c3d_train.py -h`

### Launch script

Example command to launch script

`python c3d_train.py --exp_id my_first_launch --batch_size 8 --criterion CE --display_freq 10 --threads 2 --epochs 50`

## Wiki

Consider taking a look at the wiki for additional information about the code and the datasets.

# Remarks

## Notes to self

Consider using a symbolic link if folder is already somewhere on the computer

`ln -s path/to/UCI-EGO data/UCI-EGO`
