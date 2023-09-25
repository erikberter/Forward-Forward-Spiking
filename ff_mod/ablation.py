from ff_mod.trainer import Trainer

from ff_mod.overlay import AppendToEndOverlay
from ff_mod.loss.loss import Base0Loss

from ff_mod.networks.prebuilt.ann import Base_A1_LDS as A1_LDS_ANN
from ff_mod.networks.prebuilt.snn import Base_A1_LDS as A1_LDS_SNN

from ff_mod.callbacks.accuracy_writer import AccuracyWriter
from ff_mod.callbacks.confussion_matrix import ConfussionMatrixCallback

from torch.utils.tensorboard.writer import SummaryWriter

from datetime import datetime

import logging

# Set up logging to a file
logging.basicConfig(filename=f"logs/ablation/experiments_{datetime.now().strftime('%Y%m%d%H%M%S')}.log", level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

experiments = {
    #"models": ["SNN", "ANN"],
    "models": ["SNN"],
    "depth" : [1,2,3,4,5],
    "width" : [400, 800, 1200, 2000],    
}

# Get all combinations of experiments
import itertools
keys, values = zip(*experiments.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

N_TRIALS = 5

for id, experiment in enumerate(experiments):
    if id < 19:
        continue
    
    logging.info(f"Starting experiment [{id}] {experiment}")
    PATTERN_SIZE = 100
    BATCH_SIZE = 512
    
    CLASSES = 10 # Since MNIST has 10 classes
    
    accuracy_scores = []
    
    for trial in range(N_TRIALS):
        logging.debug(f"Starting trial {trial}")
        
        # ANN
        if experiment["models"] == "ANN":
            net_non_spiking = A1_LDS_ANN(AppendToEndOverlay(100, CLASSES, 1, p=0.1), Base0Loss(threshold = 2), CLASSES, BATCH_SIZE, dims = [784 + 100, *[experiment["width"]] * experiment["depth"]], internal_epoch=10).cuda()
            
            trainer = Trainer()
            trainer.load_data_loaders(batch_size = BATCH_SIZE, test_batch_size=BATCH_SIZE)
            trainer.set_network(net_non_spiking)
            
            writer = SummaryWriter(f"out/ablation/non_spike[D{experiment['depth']}_W{experiment['width']}]_{datetime.now().strftime('%Y%m%d%H%M%S')}/" )
            trainer.add_callback(AccuracyWriter(tensorboard=writer))
            
            trainer.train(epochs=5, verbose=0)
            
            acc = trainer.test_epoch(verbose=0)
            logging.debug(f"Trial {trial}: accuracy {acc}")
            
            accuracy_scores += [acc]
        
        # SNN
        if experiment["models"] == "SNN":
            net_spiking = A1_LDS_SNN(AppendToEndOverlay(100, CLASSES, 1), Base0Loss(alpha=7, threshold = 0.3), CLASSES, BATCH_SIZE, dims = [784 + 100, *[experiment["width"]] * experiment["depth"]], internal_epoch=10).cuda()
            
            trainer = Trainer()
            trainer.load_data_loaders(batch_size = BATCH_SIZE, test_batch_size=BATCH_SIZE)
            trainer.set_network(net_spiking)
            
            writer = SummaryWriter(f"out/ablation/spike[D{experiment['depth']}_W{experiment['width']}]_{datetime.now().strftime('%Y%m%d%H%M%S')}/" )
            trainer.add_callback(AccuracyWriter(tensorboard=writer))
            
            trainer.train(epochs=5, verbose=0)
            
            acc = trainer.test_epoch(verbose=0)
            logging.debug(f"Trial {trial}: accuracy {acc}")
            
            accuracy_scores += [acc]
    
    logging.info(f"Experiment {experiment} finished with accuracy {sum(accuracy_scores)/len(accuracy_scores)}")
    accuracy_scores = []