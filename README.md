# Deep Perceptual Audio Metric (DPAM) [[Paper]](https://arxiv.org/abs/2001.04460) [[Webpage]](https://pixl.cs.princeton.edu/pubs/Manocha_2020_ADP/) [[Teaser]](https://youtu.be/dSh5SrMeyPQ) [[Full Video]](https://youtu.be/yOceeut_4Gg)

**A Differentiable Perceptual Audio Metric Learned from Just Noticeable Differences**  
[Pranay Manocha](https://www.cs.princeton.edu/~pmanocha/), [Adam Finkelstein](https://www.cs.princeton.edu/~af/), [Richard Zhang](http://richzhang.github.io/), [Nicholas J. Bryan](https://ccrma.stanford.edu/~njb/), [Gautham J. Mysore](https://ccrma.stanford.edu/~gautham/Site/Gautham_J._Mysore.html), [Zeyu Jin](https://research.adobe.com/person/zeyu-jin/)  
Accepted at [Interspeech 2020](https://arxiv.org/abs/2001.04460)

<img src='https://richzhang.github.io/index_files/audio_teaser.jpg' width=500>

This is a Tensorflow implementation (a pytorch implementation is [here](https://github.com/adrienchaton/PerceptualAudio_pytorch)) of our audio perceptual metric. It contains (0) minimal code to run our perceptual metric (DPAM), (1) code to train the perceptual metric on our JND dataset, and (2) an example of using our perceptual metric as a loss function for speech denoising.


## Also checkout our newest learned metric: CDPAM: Contrastive learning based deep perceptual audio similarity metric
Please see ```cd cdpam``` and follow the instructions in the readme. We also provide pretrained models for:
1) We make use of CDPAM for training a waveform synthesis model. We use the codebase from [here](https://github.com/descriptinc/melgan-neurips). We train both single and cross-speaker models.

2) Real-time speech enhancement: We make use of CDPAM as a loss function for training an SE model. We use the codebase from [here](https://github.com/facebookresearch/denoiser). We just augment CDPAM to their loss for training.

## Things to note:
1) At the moment, this algorithm requires using 16-bit PCM audio files to perform correctly. You can use sox to convert your file.
2) The current pretrained models support sr=22050Hz. Please make sure to resample your files first before using the metric.
For ease, you can load your audio clip using librosa.load(filename,sr=22050) and then rescale to [-32768 to 32768] using np.round(audio_file.astype(np.float)*32768). See ```metric_code/metric_use_simple.py``` for an example.

## Quick Start
Installing the metric (DPAM - perceptual audio similarity metric)
```bash
pip install dpam
```

Using the metric is as simple as: 
```bash
import dpam
loss_fn = dpam.DPAM()
wav_ref = dpam.load_audio('sample_audio/ref.wav')
wav_out = dpam.load_audio('sample_audio/2.wav')

dist = loss_fn.forward(wav_ref,wav_out)
```

## (0) Setup and basic usage

Required python libraries: Tensorflow with GPU support (>=1.14)(uses tensorflow's slim library so doesnt support tf2.0) + Scipy (>=1.1) + Numpy (>=1.14) + Tqdm (>=4.0.0). To install in your python distribution, run ```pip install -r requirements.txt```.

Additional notes:
- Warning: Make sure your libraries (Cuda, Cudnn,...) are compatible with the TensorFlow version you're using or the code will not run.
- Required software (for resampling): [SoX](http://sox.sourceforge.net/), [FFmpeg](https://www.ffmpeg.org/).
- Important note: At the moment, this algorithm requires using 32-bit floating-point audio files to perform correctly. You can use sox to convert your file.
- Tested on Nvidia GeForce RTX 2080 GPU with Cuda (>=9.2) and CuDNN (>=7.3.0). CPU mode should also work with minor changes.
- The current pretrained models support **sr=22050Hz**. Please make sure to resample your files first before using the metric.


## There are two ways to run the code:

### Using pip
This version currently supports **evaluating** the trained metric, as well as using the metric for **backpropogation(as a loss function)**. This version currently does not support a simple interface to **train** the metric. For training, please clone this repo and follow the instructions below.
```bash
pip install dpam
```
Please look at  ```example_pip.py``` as an example on how to use the function. 

### Cloning from the repository
Follow the instructions below to: 1) Evaluate the metric on a few examples, 2) Train a metric on our dataset, 3) Use the metric to optimize a downstream task and 4) Use our pretrained speech enhancement model (trained using our loss function metric).


### Minimal basic usage as a distance metric

Running the command below takes two audio files as input and gives the perceptual distance between the files. It should return (approx)**distance = 0.1929**. Some GPU's are non-deterministic, and so the distance could vary in the lsb.

```
cd metric_code
python metric_use_simple.py --e0 ../sample_audio/ref.wav --e1 ../sample_audio/2.wav
```

For loading large number of files, batch processing is more efficient. Refer to at [metric_code/metric_use.py](metric_code/metric_use.py) for more information. In short, you need to change the dataloader function `load_full_data_list()`. You also need to provide the path of the trained model as an input argument.

### Navigating this repository

**PercepAudio** - main directory
 - **metric_code** - Section 1, training our metric on our JND dataset
 - **se_code** - Section 2, training a speech enhancement model using our metric, trained above
 - **dataset** - our JND framework and dataset and text files containing perceptual judgments
 - **pre-model** - sample pre-trained models for easy reference
 - **sample_audio** - sample audio files for comparison
 -  **create_space** - sample code for creating perturbations 

## (1) Train a perceptual metric on our JND dataset

- **PercepAudio** (main directory)
   - **metric_code** 
      - main.py (train the loss function)
      - metric_use.py (use the trained model as a metric)
      - dataloader.py (collect and load the audio files)
      - helper.py (misc helper functions)
      - network_model.py (NN architecture)
      - *summaries* folder to store the new trained model with tensorboard files

### Download the JND Dataset

Go to [link](http://percepaudio.cs.princeton.edu/icassp2020_perceptual/audio_perception.zip) to download the dataset (about 23GB). After downloading the dataset, unzip the dataset into the project folder *'PerceptualAudio/dataset'*. Here are the steps to be followed:

```
git clone https://github.com/pranaymanocha/PerceptualAudio.git
cd PerceptualAudio/dataset
unzip audio_perception.zip
```

More information on the JND framework can be found in the paper [here](https://arxiv.org/abs/2001.04460). The text files in the subfolder *dataset* contain information about human perceptual judgments. This sets up the dataset for training the loss function.

For using a custom dataset, you need to follow the following steps:
1. Follow a similar framework to obtain human perceptual judgments and store them in the *dataset* subdirectory. Also create a text file containing the results of all human perceptual judgments using a convention *reference_audio_path \t noisy_audio_path \t human judgment(same(0)/different(1))*.
For an example, please see any text file in *dataset* subdirectory. 
2. Make changes to the dataloader.py function to reflect the new name/path of the folders/text file. 
3. Run the main.py function (after selecting the most appropriate set of parameters). 

Once you train a model, you can use the trained model to infer the distances between audio recordings.

### Using the trained metric for eval
You can use one of our trained models as a metric. You can also use your own trained loss function as a metric for evaluation.

For using a custom dataset, you need to follow the following steps:
1. Make sure that you have all the right requirements as specified in the *requirements.txt* file on the repo.
2. Look at *metric_use.py* for more information on how to use the trained model to infer distances between audio files. In short, you need to change the dataloader function (namely function *load_full_data_list()*). You also need to provide the path of the trained model as an input argument. Please look at metric_use.py for full information. 


### Pretrained Model
“Off-the-shelf” deep network embeddings have been used as an effective training objective that have been shown to correlate well with human perceptual judgments in the vision setting, even without being
explicitly trained on perceptual human judgments. We first investigate if similar trends hold in the audio setting. Hence, we first train a model on two audio datasets: Acoustic scene classification and Domestic audio tagging tasks of the [DCASE 2016 Challenge](https://www.cs.tut.fi/sgn/arg/dcase2016/). We keep the architecture of the model same to compare between different training regimes. More information on training this pretrained "off-the-shelf" model can be found in [this](https://github.com/francoisgermain/SpeechDenoisingWithDeepFeatureLosses) repo.

### Final Summary of the models (more info in the paper)
1. **pretrained** - pretrained "off-the-shelf" model
2. **linear** -  training linear layers over the pretrained "off-the-shelf" model
3. **finetune** - loading the pretrained "off-the-shelf weights" but training both the linear layer and the bulk model
4. **scratch** - training the full model from randomly initialized weights.  

## (2) Speech denoising with our perceptual metric as a loss function

As an application for our loss function, we use the trained loss function to train a Speech Enhancement Model. We use the Edinburgh Datashare publicly available dataset [here](https://datashare.is.ed.ac.uk/handle/10283/2791). We use the same dataset with any alteration except resampling the dataset at 16KHz.
Direct links to download the dataset and resampling can be found [here](https://github.com/francoisgermain/SpeechDenoisingWithDeepFeatureLosses). Follow the instructions to download the SE data in the script *download_sedata.sh* in the above repo. In general, we follow the same directory structure.

**PercepAudio** (main directory)
 - **se_code** (train a SE model using the above trained loss function)
    - se_train.py (train the SE system)
    - se_infer.py (Infer the SE system)
    - *dataset*
      - trainset_clean
      - trainset_noisy
      - valset_clean
      - valset_noisy
    - data_import.py (Dataloading)
    - network_model.py (NN architecture)
    - *summaries* folder to store the new trained model with tensorboard files
      
After you download the dataset, follow this directory structure to copy the audio files accordingly.

### Training the SE Model
We make use of our metric as a loss function for training an SE model. After you have downloaded the noisy dataset above (and kept the files at the correct locations), you can start training by running the command:
```
python se_train.py --args.....
```
The trained model is stored under the *summaries* folder under the folder name which you specify as an argument. The model is saved as *se_model_'+str(seconds)+'.ckpt'* where seconds is the time in seconds since epoch so that the training can be easily monitered.

### Inferring the SE Model
After you train a SE model, you can use the same trained model to denoise audio files. Simply run 
```
python se_infer.py --args....
```
with a suitable set of arguements. The denoised files will be stored in the folder name which you specify as an argument in the script. As the SE model is big, it takes a couple of hours to run on a CPU and less than 5 minutes on a GPU.

### Citation

If you use our code for research, please use the following to cite.

```
@inproceedings{Manocha:2020:ADP,
   author = "Pranay Manocha and Adam Finkelstein and Richard Zhang and Nicholas J.
      Bryan and Gautham J. Mysore and Zeyu Jin",
   title = "A Differentiable Perceptual Audio Metric Learned from Just Noticeable
      Differences",
   booktitle = "Interspeech",
   year = "2020",
   month = oct
}
```

### License
The source code is published under the [MIT license](https://choosealicense.com/licenses/mit/). See LICENSE for details. In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us. The primary contact is Pranay Manocha.<br/>
