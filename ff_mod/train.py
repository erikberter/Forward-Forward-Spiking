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

# Parameters
# TODO Add parameters injectable through console arguments
PATTERN_SIZE = 100
BATCH_SIZE = 512

CLASSES = 10 # Since MNIST has 10 classes
N_VECTORS = 1 # TODO Future: Test for more than one vector per class

# ANN

print("Training ANN")
net_non_spiking = A1_LDS_ANN(AppendToEndOverlay(PATTERN_SIZE, CLASSES, N_VECTORS, p=0.1, device='cpu'), Base0Loss(threshold = 2), 10)

trainer = Trainer(device = 'cpu')
trainer.load_data_loaders(batch_size = BATCH_SIZE, test_batch_size=BATCH_SIZE)
trainer.set_network(net_non_spiking)

writer = SummaryWriter(f"out/train/A1_LDS_BN_ANN_appendend_p34_{datetime.now().strftime('%Y%m%d%H%M%S')}/" )
trainer.add_callback(AccuracyWriter(tensorboard=writer))
trainer.add_callback(LayerIntensityCallback(net_non_spiking, tensorboard=writer))

trainer.train(epochs=7, verbose=1)

net_non_spiking.save_network("models/train/ANN_append_BN_Epoch4_B0L_T2_p34")


# SNN
print("Training SNN")
net_spiking = A1_LDS_SNN(AppendToEndOverlay(PATTERN_SIZE, CLASSES, N_VECTORS, p=0.1, device="cpu"), Base0Loss(alpha=5, beta=9, threshold = 0.25), 10)

trainer = Trainer(device = 'cpu')
trainer.load_data_loaders(batch_size = BATCH_SIZE, test_batch_size=BATCH_SIZE)
trainer.set_network(net_spiking)

writer = SummaryWriter(f"out/train/A1_LDS_SNN_BN_appendend_p34_{datetime.now().strftime('%Y%m%d%H%M%S')}/" )
trainer.add_callback(AccuracyWriter(tensorboard=writer))

trainer.train(epochs=2, verbose=1)


net_spiking.save_network("models/train/SNN_append_BN_Epoch4_B0L_T0.25_A5B7_p34")