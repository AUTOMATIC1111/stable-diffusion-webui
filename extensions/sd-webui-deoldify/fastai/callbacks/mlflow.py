"A `Callback` that saves tracked metrics and notebook file into MLflow server."
from ..torch_core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback
#This is an optional dependency in fastai.  Must install separately.
try: import mlflow
except: print("To use this tracker, please run 'pip install mlflow'")

class MLFlowTracker(LearnerCallback):
    "A `TrackerCallback` that tracks the loss and metrics into MLFlow"
    def __init__(self, learn:Learner, exp_name: str, params: dict, nb_path: str, uri: str = "http://localhost:5000"):
        super().__init__(learn)
        self.learn,self.exp_name,self.params,self.nb_path,self.uri = learn,exp_name,params,nb_path,uri
        self.metrics_names = ['train_loss', 'valid_loss'] + [o.__name__ for o in learn.metrics]

    def on_train_begin(self, **kwargs: Any) -> None:
        "Prepare MLflow experiment and log params"
        self.client = mlflow.tracking.MlflowClient(self.uri)
        exp = self.client.get_experiment_by_name(self.exp_name)
        self.exp_id = self.client.create_experiment(self.exp_name) if exp is None else exp.experiment_id
        run = self.client.create_run(experiment_id=self.exp_id)
        self.run = run.info.run_uuid
        for k,v in self.params.items():
            self.client.log_param(run_id=self.run, key=k, value=v)

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Send loss and metrics values to MLFlow after each epoch"
        if kwargs['smooth_loss'] is None or kwargs["last_metrics"] is None: return
        metrics = [kwargs['smooth_loss']] + kwargs["last_metrics"]
        for name, val in zip(self.metrics_names, metrics):
            self.client.log_metric(self.run, name, np.float(val), step=epoch)
        
    def on_train_end(self, **kwargs: Any) -> None:  
        "Store the notebook and stop run"
        self.client.log_artifact(run_id=self.run, local_path=self.nb_path)
        self.client.set_terminated(run_id=self.run)
