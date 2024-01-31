from fastai.basic_train import Learner, LearnerCallback
from fastai.vision.gan import GANLearner


class GANSaveCallback(LearnerCallback):
    """A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`."""

    def __init__(
        self,
        learn: GANLearner,
        learn_gen: Learner,
        filename: str,
        save_iters: int = 1000,
    ):
        super().__init__(learn)
        self.learn_gen = learn_gen
        self.filename = filename
        self.save_iters = save_iters

    def on_batch_end(self, iteration: int, epoch: int, **kwargs) -> None:
        if iteration == 0:
            return

        if iteration % self.save_iters == 0:
            self._save_gen_learner(iteration=iteration, epoch=epoch)

    def _save_gen_learner(self, iteration: int, epoch: int):
        filename = '{}_{}_{}'.format(self.filename, epoch, iteration)
        self.learn_gen.save(filename)
