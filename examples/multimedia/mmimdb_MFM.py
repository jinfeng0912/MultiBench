import torch
import sys
import os

sys.path.append(os.getcwd())

from utils.helper_modules import Sequential2
from unimodals.common_models import Linear, MLP, MaxOut_MLP
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat
from objective_functions.objectives_for_supervised_learning import MFM_objective
from objective_functions.recon import sigmloss1d
from training_structures.Supervised_Learning import train, test

filename = "best_mfm.pt"
traindata, validdata, testdata = get_dataloader(
    "/mnt/e/Laboratory/datasets/MM-IMDb/multimodal_imdb.hdf5", "/mnt/e/Laboratory/datasets/MM-IMDb/mmimdb",
    vgg=True, batch_size=64, num_workers=0, max_train=12000, max_val=3000, max_test=3000, no_robust=True)

classes = 23
n_latent = 512
fuse = Sequential2(Concat(), MLP(2*n_latent, n_latent, n_latent//2)).cuda()
encoders = [MaxOut_MLP(512, 512, 300, n_latent, False).cuda(
), MaxOut_MLP(512, 1024, 4096, n_latent, False).cuda()]
head = Linear(n_latent//2, classes).cuda()

decoders = [MLP(n_latent, 600, 300).cuda(), MLP(n_latent, 2048, 4096).cuda()]
intermediates = [MLP(n_latent, n_latent//2, n_latent//2).cuda(),
                 MLP(n_latent, n_latent//2, n_latent//2).cuda()]

recon_loss = MFM_objective(2.0, [sigmloss1d, sigmloss1d], [
                           1.0, 1.0], criterion=torch.nn.BCEWithLogitsLoss())

train(encoders, fuse, head, traindata, validdata, 10, decoders+intermediates, early_stop=False, task="multilabel",
      objective_args_dict={"decoders": decoders, "intermediates": intermediates}, save=filename, optimtype=torch.optim.AdamW, lr=5e-3, weight_decay=0.01, objective=recon_loss)

print("Testing:")
model = torch.load(filename).cuda()
test(model, testdata, method_name="MFM", dataset="imdb",
     criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel", no_robust=True)
