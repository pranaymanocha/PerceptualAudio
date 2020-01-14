# Speech Denoising

As an application for our loss function, we use the trained loss function to train a Speech Enhancement Model. We use the Edinburgh Datashare publicly available dataset [here](https://datashare.is.ed.ac.uk/handle/10283/2791). We use the same dataset with any alteration except resampling the dataset at 16KHz.
Direct links to download the dataset and resampling can be found [here](https://github.com/francoisgermain/SpeechDenoisingWithDeepFeatureLosses). Follow the instructions to download the SE data in the script *download_sedata.sh* in the above repo. In general, we follow the same directory structure.

- **PercepAudio** (main directory)
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

### Training the SE Model:
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


## License
[MIT](https://choosealicense.com/licenses/mit/)
