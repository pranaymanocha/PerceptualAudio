# SE Enhancement Code

Using the codebase for speech enhancement using the trained loss function

## Installation

Use the same requirements as mentioned in "requirements.txt" on the homepage

## Usage

Create a folder "dataset" inside se_code where you should have the SE dataset 
- trainset_clean
- trainset_noisy
- valset_clean
- valset_noisy

(Follow [this]((https://github.com/francoisgermain/SpeechDenoisingWithDeepFeatureLosses)) for how to download and resample the dataset. Resample the dataset to 16KHz).

The se_infer.py file needs 2 arguments:
1) path of the SE model to be loaded. (ex - ../pre-model/se_model/se_model.ckpt)
2) actual model name that you are loading. (for ex - m1)

After you specify these values, do 
```python
python se_infer.py --model_folder path  --model_name m1 
```
After you run this command, the files will be created in the directory "dataset" under the name as "dataset/valset_noisy_"model_name"_denoised/filename.wav

## License
[MIT](https://choosealicense.com/licenses/mit/)
