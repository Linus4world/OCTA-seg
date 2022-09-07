from ray import tune
from ray.tune.schedulers.pb2 import PB2
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.hyperopt import HyperOptSearch
import hyperopt.hp
import ConfigSpace as CS

class OPTIMIZE_TYPE():
    PB2 = "PB2"
    ASHA = "ASHA"
    BOHB = "BOHB"


def get_raytune_config(type: OPTIMIZE_TYPE, metric: str, mode="max", seed=0):
    """
    The function currently supports three optimization types:
        - BOHB: Bayesian Optimization HyperBand (2018) (available at https://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf)
                The BOHB used Bayesian inference to select promising hyperparamters. It can optimize continous and descrete paramaters as well as different architectures.
        - ASHA: Asyncronic HyperBand Scheduler (2020) (available at https://arxiv.org/pdf/1810.05934.pdf)
                ASHA builds on top of BOHB but terminates bad trails early by a halfing rate every `reduction_rate` steps. This is computationally more efficient but 
                might terminate trails that would have become great later in training. It can optimize continous and descrete paramaters as well as different architectures.
        - PB2: Population based bandits (2020) (available at https://proceedings.neurips.cc/paper/2020/file/c7af0926b294e47e52e46cfebe173f20-Paper.pdf)
                PB2 is a more efficient implementation of Population Based Training (PBT) yields good results even with a small populatio size of 4 actors.
                Different from the other schedulers, PB2 creates a hyperparamter schedule, i.e. what is the best parameter at a given time step.
                Bad trails are terminated configuration from good trails is exploited. PB2 makes heavily use of checkpoints.
                It can optimize continous and descrete paramaters but because of its exploitation step, it cannot be used with different architectures.
    """
    if type == OPTIMIZE_TYPE.PB2:
        search_space = {
            "lr": [0.01, 0.00001],
            "batch_size": [1,16],
        }
        start_search_space = {
            "lr": tune.uniform(0.001, 0.0001),
            "batch_size": tune.choice([4,8,16]),
        }
        scheduler = PB2(
            time_attr="training_iteration",
            metric=metric,
            mode=mode,
            perturbation_interval=5,
            hyperparam_bounds=search_space,
            start_search_space=start_search_space
        )
        search_alg = None
        num_samples = 4
    elif type == OPTIMIZE_TYPE.ASHA:
        scheduler = AsyncHyperBandScheduler(
            time_attr='training_iteration',
            metric=metric,
            mode=mode,
            grace_period=50,
            max_t=300,
            reduction_factor=25
        )
        param_space = {
            "lr": hyperopt.hp.quniform("lr",1e-5, 1e-2, 1e-5),
            "batch_size": hyperopt.hp.choice("batch_size", [4,8,16]),
            "model": hyperopt.hp.choice("model",["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]),
            "loss": hyperopt.hp.choice("loss", ["CosineEmbeddingLoss"])
        }
        start_config = [{
            "lr": 0.001,
            "batch_size": 8,
            "model": "efficientnet_b1",
            "loss": "CosineEmbeddingLoss"
        }]
        search_alg = HyperOptSearch(
            space=param_space,
            metric=metric,
            mode=mode,
            random_state_seed=seed,
            points_to_evaluate=start_config
        )
        num_samples = 32

    elif type == OPTIMIZE_TYPE.BOHB:
        scheduler = HyperBandForBOHB(
            metric=metric,
            mode=mode,
            max_t=300,
            reduction_factor=25
        )
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter("lr", lower=1e-5, upper=1e-2))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter("batch_size", choices=[4,8,16]))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter("model", choices=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter("loss", choices=["CosineEmbeddingLoss"]))
        start_config = [{
            "lr": 0.001,
            "batch_size": 8,
            "model": "efficientnet_b1",
            "loss": "CosineEmbeddingLoss"
        }]
        search_alg = TuneBOHB(
            space=config_space,
            metric=metric,
            mode=mode,
            points_to_evaluate=start_config,
            seed=seed
        )
        num_samples=32
    else:
        raise NotImplementedError(type)


    return type, scheduler, search_alg, num_samples
