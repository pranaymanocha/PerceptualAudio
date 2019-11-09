# A DIFFERENTIABLE PERCEPTUAL AUDIO METRIC LEARNED FROM JUST NOTICEABLE DIFFERENCES

This is a Tensorflow implementation of our [paper](https://percepaudio.cs.princeton.edu/icassp2020_perceptual/).

Contact: [Pranay Manocha](https://www.cs.princeton.edu/~pmanocha/)

# Setup

Required python libraries: Tensorflow with GPU support (>=1.13) + Scipy (>=1.1) + Numpy (>=1.14) + Tqdm (>=4.0.0). To install in your python distribution, run

```bash
pip install -r requirements.txt
```
Warning: Make sure your libraries (Cuda, Cudnn,...) are compatible with the TensorFlow version you're using or the code will not run.

Required software (for resampling): [SoX](http://sox.sourceforge.net/)

Important note: At the moment, this algorithm requires using 32-bit floating-point audio files to perform correctly. You can use sox to convert your file.

Tested on Nvidia GeForce RTX 2080 GPU with Cuda (>=9.2) and CuDNN (>=7.3.0). CPU mode should also work with minor changes.

## Navigating this repository

There are two main sections:
1. Training a loss function
2. Using this trained loss function to train a SE model

Section 1 is found in the "code" subfolder of this repo, whereas Section 2 is in the "se_code" subfolder.

# Section 1 - Train a loss function

### Download the JND Dataset

Go to [link](audio_files.zip) to download the dataset. (Warning) The zip file is about 23GB. After downloading the dataset, unzip the dataset into the project folder 'PerceptualAudio/dataset_collection'. Here are the steps to be followed:

```python
git clone https://github.com/pranaymanocha/PerceptualAudio.git
cd PerceptualAudio/dataset_collection
wget audio_files.zip
unzip audio_files.zip
```
More information on the JND framework can be found [here](some link). The text files in the subfolder dataset_collection contain information about human perceptual judgments. This sets up the dataset for training the loss function.

For using a custom dataset, you need to follow a similar framework to first obtain human perceptual judgments and then creating a folder containing the audio files in a consistent naming convention. You then need to create a text file we created above in a similar fashion like "ref_audio_path \t noisy_audio_path \t human_judgement{0,1}" where {0,1} depict the person answered same or different respectively. Change the function in the dataloader to reflect these changes to the text files and run the main.py function. After you train a model, you can use the trained model to infer the distances between audio recordings.

### Using the trained metric for eval:
You can use one of our trained models as a metric. You can also use your own trained loss function as a metric for evaluation.

Make sure that you have all the correct requirements. See the example code file "metric_use.py" inside the "code" folder. The sample code explains on how to use the loss models for evaluation. In short, while evaluating on your own custom dataset, you need to change the data-loader function to load data, as well as provide a new path for the new trained model weights. Look at "metric_use.py" for more information.

## Speech Denoising
Follow the instructions in [this link](https://github.com/francoisgermain/SpeechDenoisingWithDeepFeatureLosses) on downloading the noisy dataset from Edinburgh DataShare. We follow the same directory structure as given the link above. Noisy data
The data used to train and test our system is available publicly on the Edinburgh DataShare website at https://datashare.is.ed.ac.uk/handle/10283/2791. Information on how the dataset is constructed can be found in Valentini-Botinhao et al., 2016. The dataset was used without alteration except for resampling at 16kHz.
Data directory structure:

- PercepAudio
  - code
  - se_code 
    - dataset
      - trainset_noisy
      - trainset_clean
      - valset_clean
      - valset_noisy

### Training the SE Model:
We make use of our metric as a loss function for training a SE model. After you have downloaded the noisy dataset above (and kept the files at the correct locations), you can start training by running the command:
```python
python se_train.py --args.....
```
The trained model is stored under the *summaries* folder under the folder which you specify it into. The model is saved as *se_model_'+str(seconds)+'.ckpt'* where seconds is the time in seconds since Epoch. This convention is chosen so that the training of the SE model can be regularly monitered. 

### Inferring the SE Model
After you train a SE model, you can use the same trained model to denoise audio files. Simply run 
```python
python se_infer.py --model_folder path1 --model_name name1
```
with the set of parameters. The denoised files will be stored in the folder name which you specify as an argument in the script As the SE model is big, it takes a couple of hours to run on a CPU and less than 5 minutes on a GPU.

## Pretrained Network
Data

Our feature loss network is trained on the acoustic scene classification and domestic audio tagging tasks of the [DCASE 2016 Challenge(https://www.cs.tut.fi/sgn/arg/dcase2016/). 
Follow [this repo] for steps on what/how to train this network. 

## Citation
If you use our code for research, please cite our paper:
## Licence 
The source code is published under the MIT license. See LICENSE for details. In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us.
