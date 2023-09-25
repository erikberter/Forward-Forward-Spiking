from ff_mod.trainer import Trainer

from ff_mod.overlay import corner_overlay, AppendToEndOverlay
from ff_mod.loss.loss import Base0Loss, ContrativeFF

from ff_mod.networks.prebuilt.ann import A1_LDS_ANN
from ff_mod.networks.prebuilt.snn import A1_LDS as A1_LDS_SNN

from ff_mod.callbacks.accuracy_writer import AccuracyWriter
from ff_mod.callbacks.confussion_matrix import ConfussionMatrixCallback
from ff_mod.callbacks.ff_layer_intensity import LayerIntensityCallback

from torch.utils.tensorboard.writer import SummaryWriter

from datetime import datetime

import torch
import numpy as np


torch.manual_seed(0)
np.random.seed(0)

# Parameters
# TODO Add parameters injectable through console arguments
PATTERN_SIZE = 100
BATCH_SIZE = 512

CLASSES = 10 # Since MNIST has 10 classes
N_VECTORS = 1 # TODO Future: Test for more than one vector per class

for size in [400, 800]:
    for dataset in ["emnist"]:

        # ANN

        print("Training ANN")
        net_non_spiking = A1_LDS_ANN(AppendToEndOverlay(PATTERN_SIZE, 27, N_VECTORS, p=0.1), Base0Loss(threshold = 2), 27, mid_size=size).cuda()

        trainer = Trainer(device = 'cuda:0')
        trainer.load_data_loaders(dataset, batch_size = BATCH_SIZE, test_batch_size=BATCH_SIZE)
        trainer.set_network(net_non_spiking)

        writer = SummaryWriter(f"out/train_paper2/{dataset}_LDS_ANN({size})_appendToEnd_{datetime.now().strftime('%Y%m%d%H%M%S')}/" )
        trainer.add_callback(AccuracyWriter(tensorboard=writer))

        trainer.train(epochs=10, verbose=1)

        net_non_spiking.save_network(f"models/train_paper2/{dataset}_ANN({size})_appendToEnd_Epoch10_B0L_T2_p1")


        # SNN
        print("Training SNN")
        net_spiking = A1_LDS_SNN(AppendToEndOverlay(PATTERN_SIZE, 27, N_VECTORS, p=0.1), Base0Loss(alpha=15, beta=19, threshold = 0.25), 27, mid_size=size).cuda()


        trainer = Trainer(device = 'cuda:0')
        trainer.load_data_loaders(dataset, batch_size = BATCH_SIZE, test_batch_size=BATCH_SIZE)
        trainer.set_network(net_spiking)

        writer = SummaryWriter(f"out/train_paper2/{dataset}_LDS_SNN({size})_appendToEnd_{datetime.now().strftime('%Y%m%d%H%M%S')}/" )
        trainer.add_callback(AccuracyWriter(tensorboard=writer))

        trainer.train(epochs=10, verbose=1)


        net_spiking.save_network(f"models/train_paper2/{dataset}_SNN({size})_appendToEnd_Epoch10_B0L_T0.25_A5B9_p1")