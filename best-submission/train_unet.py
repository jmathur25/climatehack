# %%
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, ConcatDataset
import xarray as xr
import wandb

# custom
from utils.loss import MS_SSIMLoss
from utils.data import ClimatehackDataset, CustomDataset

# %%
from fastai.vision.all import *
from fastai.callback.wandb import *
from fastai.callback.tracker import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *
from fastai.optimizer import ranger
from fastai.layers import Mish

# %%
NAME = "xse_resnext50_deeper"
BATCH_SIZE = 32
FORECAST = 24

wandb.init(project="climatehack", group=NAME)
# %%
# train_ds = ClimatehackDataset("data/data.npz", with_osgb=True)
train_ds = CustomDataset("../data/train.npz")
valid_ds = CustomDataset("../data/test.npz")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

dls = DataLoaders(train_loader, valid_loader, device=torch.device("cuda"))
# %%
criterion = MS_SSIMLoss(channels=FORECAST, crop=True)

# %%
arch = partial(xse_resnext50_deeper, act_cls=Mish, sa=True)
model = create_unet_model(
    arch=arch,
    img_size=(128, 128),
    n_out=24,
    pretrained=False,
    n_in=train_ds[0][0].shape[0],
    self_attention=True,
)

# %%
callbacks = [
    SaveModelCallback(monitor="train_loss", fname=NAME),
    ReduceLROnPlateau(monitor="train_loss", factor=2),
    # EarlyStoppingCallback(monitor="val_loss", patience=10, mode="min"),
    WandbCallback(),
]
learn = Learner(
    dls,
    model,
    loss_func=criterion,
    cbs=callbacks,
    model_dir="checkpoints",
    opt_func=ranger,
)

# %%
learn.fit_flat_cos(100, 1e-3)
