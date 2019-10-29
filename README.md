# PerceptualAudio
A DIFFERENTIABLE PERCEPTUAL AUDIO METRIC LEARNED FROM JUST NOTICEABLE DIFFERENCES

This is a Tensorflow implementation of our paper

# Project Title

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc



Contact: Pranay Manocha


Citation
If you use our code for research, please cite our paper: . Speech Denoising with Deep Feature Losses. arXiv:1806.10522. 2018.

License
The source code is published under the MIT license. See LICENSE for details. In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us.

Setup

Requirement

Required python libraries: Tensorflow with GPU support (>=1.4) + Scipy (>=1.1) + Numpy (>=1.14) + Tqdm (>=4.0.0). To install in your python distribution, run

pip install -r requirements.txt

Warning: Make sure your libraries (Cuda, Cudnn,...) are compatible with the tensorflow version you're using or the code will not run.

Required software (for resampling): SoX (Installation instructions)

Important note: At the moment, this algorithm requires using 32-bit floating-point audio files to perform correctly. You can use sox to convert your file. To convert audiofile.wav to 32-bit floating-point audio at 16kHz sampling rate, run:

sox audiofile.wav -r 16000 -b 32 -e float audiofile-float.wav

Tested in Ubuntu + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes.


Default data download
In order to run our algorithm with default parameters, you need to download the noisy dataset from Edinburgh DataShare (see below). The dataset can be automatically downloaded and pre-processed (i.e. resampled at 16kHz) by running the script

./download_sedata.sh

To download only the testing data, you can run the reduced script:

./download_sedata_onlyval.sh

Using custom data
If you want to use your own data for testing, you need to put all the .wav files in a single folder.

If you want to use your own data for training, you need to put your data in a single top folder. In that folder, you should have 4 individual folders:

trainset_noisy/ (for the noisy speech training files),
trainset_clean/ (for the ground truth clean speech training files),
valset_noisy/ (for the noisy validation files), and
valset_clean/ (for the noisy validation files).
The validation folders may be empty but they must exist. Matching files in the corresponding noisy and clean folders must have the same name.

The audio data must be sampled at 16kHz (you can resample your data using SoX - see download_data.sh for an example).

Top

Denoising scripts
Testing with default parameters
Once you've downloaded in the script download_data.sh, you can directly process the testing dataset by running

python senet_infer.py

The denoised files will be stored in the folder dataset/valset_noisy_denoised/, with the same name as the corresponding source files in dataset/valset_noisy/.

In our configuration, the algorithm allocates ~5GB of memory on the GPU for training. Running the code as is on GPUs with less memory may fail.

Testing with custom data and/or denoising model
If you have custom testing data (formatted as described above) stored in a folder foldername/ and/or a custom denoising model with names se_model.ckpt.* stored in a folder model_folder/, you can test that model on that data by running:

python senet_infer.py -d folder_name -m model_folder

The denoised files will be stored in the folder folder_name_denoised/, with the same name as the corresponding source files.

Warning: At this time, when using a custom model, you must make sure that the system parameters in senet_infer.py match the ones used in the stored denoising model or the code won't run properly (if running at all).

Training with default parameters
Once you've downloaded in the script download_data.sh, you can directly train a model using the training dataset by running

python senet_train.py

The trained model will be stored in the root folder with the names se_model.ckpt.*.

In our configuration, the algorithm allocates ~5GB of memory on the GPU for training. Running the code as is on GPUs with less memory may fail.

Training with custome data and/or feature loss model
If you have custom training data (formatted as described above) stored in a folder foldername/ and/or a custom feature loss model with names loss_model.ckpt.* stored in a folder loss_folder/, you can train a speech denoising model on that data using that feature loss model by running:

python senet_train.py -d folder_name -l loss_folder -o out_folder

The trained model will be stored in folder out_folder/ (default is root folder) with the names se_model.ckpt.*.

Warning: At this time, when using a custom loss model, you must make sure that the system parameters in senet_train.py match the ones used in the stored loss model or the code won't run properly (if running at all).

Top


Models
The deep feature loss network graph and parameters are stored in the models/loss_model.ckpt.* files.

The denoising network graph and parameters are stored in the models/se_model.ckpt.* files. This model was trained following the procedure described in our associated paper. The current training script se_train.py is parameterized in such a way that an identical training procedure as in our associated paper would be performed on the specified training dataset.

Top


Noisy data
The data used to train and test our system is available publicly on the Edinburgh DataShare website at https://datashare.is.ed.ac.uk/handle/10283/2791. Information on how the dataset is constructed can be found in Valentini-Botinhao et al., 2016. The dataset was used without alteration except for resampling at 16kHz.

Top


Deep feature loss training
We also provide scripts to (re-)train the loss model. As of know, using the two classification tasks described in our paper is hard-coded.

Data
Our feature loss network is trained on the acoustic scene classification and domestic audio tagging tasks of the [DCASE 2016 Challenge(https://www.cs.tut.fi/sgn/arg/dcase2016/). Downloading and pre-processing (i.e., downsampling to 16kHz) the corresponding data can be done by running the script:

./download_lossdata.sh

Warning: The training script expects the data at the locations set in the downloading script.

Top

Training script
Once the data is downloaded, you can (re-)train a deep feature loss model by running:

python lossnet_train.py

The loss model is stored in the root folder by default. A custom output directory for loss model can be specified as:

python lossnet_train.py -o out_folder

Top


Notes
Currently, the download scripts are only provided for UNIX-like systems (Linux & Mac OSX). If you plan on running our algorithm on Windows, please contact us and/or download and resample the data "by hand".

Currently, dilation for 1-D layers is not properly implemented in the Tensorflow slim library we use. The functions signal_to_dilated and _dilated_to_signal in helper.py allows to transform a 1-D layer into an interlaced 2-D layer such that undilated convolution on the 2-D layer is equivalent to dilated convolution on the 1-D layer.

Top


SoX installation instructions
The latest version of SoX can be found on their SourceForge page at https://sourceforge.net/projects/sox/files/sox/ (Go to the folder corresponding to the latest version). Below are additional details regarding the installations for many common operating systems.

Linux
Ubuntu
As of June 13, 2018, SoX can be installed from the Ubuntu repositories by running in a terminal:

sudo apt-get install sox

Fedora
As of June 13, 2018, SoX can be installed from the Fedora repositories by running in a terminal:

sudo yum install sox

Mac OSX
Homebrew
If you have Homebrew installed, just run in a terminal:

brew install sox

Macports
If you have Macports installed, just run in a terminal:

port install sox

You may need to run the command with root priviledges, in which case, run in a terminal:

sudo port install sox

Pre-compiled version
SoX provides a pre-compiled executable for Mac OSX. You can download it at https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-macosx.zip/download.

Then unzip the downloaded archive and move the extracted folder to your Applications folder.

The last step is to add that folder to your path. To do so, run in a terminal:

cd ~
echo "" >> .bash_profile
echo "# Adding SoX to path" >> .bash_profile
echo "export PATH=\$PATH:/Applications/sox-14.4.1" >> .bash_profile
source .bash_profile
Warning: The executable hasn't been updated since 2015 so consider using one of the two options above instead or compile from sources if the executable fails

Install from sources (Unix-like systems)
Download sources from the terminal using:

wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz/download

Un-compress the archive:

tar -zxvf sox-14.4.2.tar.gz

Go into the folder with the extracted files:

cd sox-14.4.2

Compile and install SoX:

./configure
make -s
make install
Warning: Make sure there are no space in any of the folder name on the path of the source files or the building will fail.

Windows
Follow instructions provided here. If you need additional assistance, please contact us.

Top
