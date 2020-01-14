# Perceptual Metrics of Audio JND Dataset

## Citation


## Downloading the dataset
The link for downloading the link is [here ~30G](http://percepaudio.cs.princeton.edu/icassp2020_perceptual/audio_perception.zip). This downloads the zip file. Please unzip the zip file into the *dataset* folder in the main directory.

## JND Framework
Please look at the paper [here](http://arxiv.org/abs/2001.04460) to know more about the active learning we employ to collect JND dataset from Turk study.

## Instructions to use
JND Audio pairwise dataset. The database was designed to train an audio loss metric that takes in two audio files and tells if the two files are the same or different.
 
The audio files are sampled at 48kHz. This dataset consists of 4 versions which lead to the development of the final dataset:
1) **v1**:

Linear Noises: choose one of these linear noises:  applause,blue_noise,brown_noise,crickets,pink_noise,siren,violet_noise,water_drops, white_noise, reverb_noise and mp3 noise.

File **dataset_linear.txt** contains the paths and links of the JND comparisons done.

2) **v2**:

Reverb Noises: consist of perturbations like DRR and RT60

File **dataset_reverb.txt** contains the paths and links of the JND comparisons done.

3) **v3**:

EQ Noises: consist of combined perturbations like varying the frequency content between different frequency bands like 500Hz and 1kHz.

File **dataset_eq.tx** contains the paths and links of the JND comparisons done.

4) **v4**:

Combined Noises: consist of perturbations like linear, reverb, compression and EQ. 

linear consists of (applause,blue_noise,brown_noise,crickets,pink_noise,siren,violet_noise,water_drops, white_noise, reverb_noise and mp3 noise)

reverb noises consists of (DRR,RT60)

compression noise consists of (MP3,mu-law)

EQ noise consists of (500Hz,1000Hz)

Choose one of each category and randomly choose an order. 

File **dataset_combined.txt** contains the paths and links of the JND comparisons done.


## License
[MIT](https://choosealicense.com/licenses/mit/)
