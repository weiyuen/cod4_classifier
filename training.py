import pytorch_lightning as pl
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from transforms import train_tfs, test_tfs
from model import PLModel


batch_size = 64
num_workers = 8
epochs = 100

train_path = r'B:\Datasets\cod4\train'
val_path = r'B:\Datasets\cod4\val'
test_path = r'B:\Datasets\cod4\test'


def main():
    train_ds = ImageFolder(train_path, transform=train_tfs)
    val_ds = ImageFolder(val_path, transform=test_tfs)
    test_ds = ImageFolder(test_path, transform=test_tfs)

    train_datagen = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )

    val_datagen = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    test_datagen = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )


    model = PLModel()
    callback = pl.callbacks.ModelCheckpoint(monitor='val_loss')

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epochs,
        callbacks=[callback],
        precision=16,
        auto_lr_find=True
    )

    trainer.tune(model, train_datagen)
    trainer.fit(model, train_datagen, val_datagen)

if __name__ == '__main__':
    main()