'''
Prediction script for CoD4 screenshots.
Prints and returns predictions for images in a given folder.
0 = Clean, 1 = Hacking
'''

import argparse
import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import PLModel
from transforms import test_tfs


class InferenceDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.transform = transform
        self.images = os.listdir(path)
        self.images = [os.path.join(path, image) for image in self.images]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        return image


def get_preds():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    # Initialize model and load trained weights.
    model = PLModel.load_from_checkpoint(r'lightning_logs/version_0/checkpoints/epoch=4-step=7685.ckpt')
    model.to(device)
    model.eval();
    
    # Create Dataset and DataLoader.
    ds = InferenceDataset(args.image_dir, transform=test_tfs)
    datagen = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    # Inference Loop.
    outputs = []
    
    for batch in tqdm(datagen):
        batch = batch.to(device)
        logits = model(batch)
        preds = torch.sigmoid(logits)
        preds = preds.to('cpu').detach().tolist()
        preds = [value for pred in preds for value in pred] # flatten list
        outputs.append(preds)

    outputs = [value for output in outputs for value in output]
    for val in outputs:
        message = "HACKER" if val > args.threshold else ""
        print(f'{val:.2f} {message}')
    
    outputs = [1 if val > args.threshold else 0 for val in outputs]
    
    return(outputs)


if __name__ == '__main__':
    get_preds()