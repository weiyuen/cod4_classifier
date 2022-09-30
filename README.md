# CoD4 Screenshot Classifier


<img src="https://user-images.githubusercontent.com/71860925/183033221-287dfe45-6ef9-4ad7-bd55-beebfda8c9e4.jpg" width="500" value="Clean"> <img src="https://user-images.githubusercontent.com/71860925/183033354-8515eda9-e843-48a3-b588-40f0f5d3d445.jpg" width="500" value="Hacking">

This repo contains training and inference code for a CNN-based CoD4 screenshot classifier to detect hackers. (Currently deployed on the [NamelessNoobs](https://namelessnoobs.com/cod4ss/public.php) servers.)

## Inference Usage:
1. Clone the repository and install required libraries using `pip install -r requirements.txt`
2. Download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1JM8ZlH1k3iFgPIjLxok3nrqSVYG1f5Pb?usp=sharing) and place the file in `lightning_logs/version_1/checkpoints`
3. Run predict_batches.py e.g. `python predict_batches.py --image_dir=path/to/folder --threshold=0.1`. The script takes the following arguments:
- `image_dir` path to the image folder
- `batch_size` inference batch size, adjust according to available memory. Default = 8.
- `threshold` probability threshold for classifying a screenshot as a hacker. Default = 0.5.
4. The script prints and returns predictions for all images, where 1 is the hacking class, and 0 is the clean class.


