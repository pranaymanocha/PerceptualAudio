# Contrastive learning-based Deep Perceptual Audio Metric (CDPAM) 

[[Real-time Speech Enhanced Examples]](https://percepaudio.cs.princeton.edu/Manocha20_CDPAM/clips/clips_se.php)

[[MelGAN Single Speaker enhanced Examples]](https://percepaudio.cs.princeton.edu/Manocha20_CDPAM/clips/clips_mel_single.php)

[[MelGAN Cross Speaker enhanced Examples]](https://percepaudio.cs.princeton.edu/Manocha20_CDPAM/clips/clips_mel_cross.php)

[Download Pretrained SE + Vocoder trained models](https://percepaudio.cs.princeton.edu/Manocha20_CDPAM/mel_se.zip)

**Contrastive Learning based Perceptual Audio Similarity**

[Pranay Manocha](https://www.cs.princeton.edu/~pmanocha/), [Zeyu Jin](https://research.adobe.com/person/zeyu-jin/), [Richard Zhang](http://richzhang.github.io/), [Adam Finkelstein](https://www.cs.princeton.edu/~af/)   

<img src='https://richzhang.github.io/index_files/audio_teaser.jpg' width=500>

This is a Pytorch implementation of our new and improved audio perceptual metric. It contains (0) minimal code to run our perceptual metric (CDPAM).

## (0) Usage as a loss function

### Minimal basic usage as a distance metric

Running the command below takes two audio files as input and gives the perceptual distance between the files. It should return (approx)**distance = 0.1696**. Some GPU's are non-deterministic, and so the distance could vary in the lsb.

Installing the metric (CDPAM - perceptual audio similarity metric)
```bash
pip install cdpam
```
Please also run ``` pip install -r requirements.txt``` to install all the requirements required to run the metric.

Using the metric is as simple as: 
```bash
import cdpam
loss_fn = cdpam.CDPAM()
wav_ref = cdpam.load_audio('sample_audio/ref.wav')
wav_out = cdpam.load_audio('sample_audio/2.wav')

dist = loss_fn.forward(wav_ref,wav_out)
```
