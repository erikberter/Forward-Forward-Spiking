from ff_mod.trainer import Trainer

from ff_mod.overlay import corner_overlay
from ff_mod.loss.loss import Base0Loss, Base1Loss, SymbaLoss, SymVarLoss, ContrativeFF

from ff_mod.networks.prebuilt.ann import A1_LDS_ANN
from ff_mod.networks.prebuilt.snn import A1_LDS as A1_LDS_SNN


from datetime import datetime

import logging

date_str = datetime.now().strftime('%Y%m%d%H%M%S')

# Set up logging to a file
logging.basicConfig(filename=f"logs/loss/experiments_{date_str}.log", level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
# TODO Add parameters injectable through console arguments

BATCH_SIZE = 512

CLASSES = 10 # Since MNIST has 10 classes




trainer = Trainer()
trainer.load_data_loaders(batch_size = BATCH_SIZE, test_batch_size=BATCH_SIZE)


def train_and_get_accuracy(loss):
    net = net_model(corner_overlay, loss, 10).cuda()
    
    trainer.set_network(net)
    
    trainer.train_epoch(break_step = 16, verbose=0)
    
    return trainer.test_epoch(verbose=0)

def base0loss_objective(trial):
    threshold_param = trial.suggest_float("threshold", 0, 100, log=True)
    #alpha_param = trial.suggest_float("alpha", 1, 100, log=True)
    #beta_param = trial.suggest_float("beta", 1, 100, log = True)

    loss = Base0Loss(threshold=threshold_param/100, alpha=5, beta=8)
    
    return train_and_get_accuracy(loss)


def base1loss_objective(trial):
    threshold_param = trial.suggest_float("threshold", 0, 0.5)
    alpha_param = trial.suggest_float("alpha", 1, 1000, log=True)
    beta_param = trial.suggest_float("beta", 1, 1000, log = True)

    loss = Base1Loss(threshold=threshold_param, alpha=alpha_param, beta=beta_param)
    
    return train_and_get_accuracy(loss)


def symbaloss_objective(trial):
    alpha_param = trial.suggest_float("alpha", 0.1, 10)

    loss = SymbaLoss(alpha=alpha_param)
    
    return train_and_get_accuracy(loss)

def symvarloss_objective(trial):
    alpha_param = trial.suggest_float("alpha", 0.1, 10)

    loss = SymVarLoss(alpha=alpha_param)
    
    return train_and_get_accuracy(loss)

def contrastiveff_objective(trial):
    threshold_param = trial.suggest_float("threshold", 0, 50, log=True)
    #alpha_param = trial.suggest_float("alpha", 0.1, 10)
    #ratio_param = trial.suggest_float("ratio", 0.01, 10)

    loss = ContrativeFF(threshold=threshold_param/100, alpha=7, ratio=6)
    
    return train_and_get_accuracy(loss)


# Optuna Training

import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt

import os

#net_models = [A1_LDS_ANN, A1_LDS_SNN]
#objectives = [base0loss_objective, base1loss_objective, symbaloss_objective, symvarloss_objective, contrastiveff_objective]

net_models = [A1_LDS_SNN]
objectives = [base0loss_objective, contrastiveff_objective]

for net_model in net_models:
    for objective in objectives:
        study = optuna.create_study(direction="maximize")


        study.optimize(objective, n_trials=100)


        best_params = study.best_params
        logging.info(f"In Model {net_model.__name__} with loss {objective.__name__} Best Parameters: {best_params}")
        
        route = f"out/loss/{date_str}/{objective.__name__}_Log/{net_model.__name__}/"
        
        if not os.path.exists(route):
            os.makedirs(route)
        
        fig = vis.plot_optimization_history(study)
        fig.write_image(route+ "plot_optimization_history.png")

        fig = vis.plot_parallel_coordinate(study)
        fig.write_image(route+ "plot_parallel_coordinate.png")

        fig = vis.plot_param_importances(study)
        fig.write_image(route+ "plot_param_importances.png")

        fig = vis.plot_slice(study)
        fig.write_image(route+ "plot_slice.png")
        
        fig = vis.plot_contour(study)
        fig.write_image(route+ "plot_contour.png")
        
        fig = vis.plot_edf(study)
        fig.write_image(route+ "plot_edf.png")
        
        fig = vis.plot_rank(study)
        fig.write_image(route+ "plot_rank.png")