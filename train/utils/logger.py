from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only
import requests, json
import os

class ClientLogger(LightningLoggerBase):
    @property
    def name(self):
        return "ClientLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        print(f"Logging metrics: {metrics}")
        print(f"Logging step: {step}")
        metrics['step'] = step
        print(f"Logging metrics: {metrics}")
        if os.environ.get('IS_LOGGER_ON'):
            requests.post(os.environ.get('LOGGER_URL', False), data=json.dumps(metrics))


    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
