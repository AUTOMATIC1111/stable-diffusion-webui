import os
import wandb
from annotator.oneformer.detectron2.utils import comm
from annotator.oneformer.detectron2.utils.events import EventWriter, get_event_storage


def setup_wandb(cfg, args):
    if comm.is_main_process():
        init_args = {
            k.lower(): v
            for k, v in cfg.WANDB.items()
            if isinstance(k, str) and k not in ["config"]
        }
        # only include most related part to avoid too big table
        # TODO: add configurable params to select which part of `cfg` should be saved in config
        if "config_exclude_keys" in init_args:
            init_args["config"] = cfg
            init_args["config"]["cfg_file"] = args.config_file
        else:
            init_args["config"] = {
                "model": cfg.MODEL,
                "solver": cfg.SOLVER,
                "cfg_file": args.config_file,
            }
        if ("name" not in init_args) or (init_args["name"] is None):
            init_args["name"] = os.path.basename(args.config_file)
        else:
            init_args["name"] = init_args["name"] + '_' + os.path.basename(args.config_file)
        wandb.init(**init_args)


class BaseRule(object):
    def __call__(self, target):
        return target


class IsIn(BaseRule):
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(self, target):
        return self.keyword in target


class Prefix(BaseRule):
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(self, target):
        return "/".join([self.keyword, target])


class WandbWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self):
        """
        Args:
            log_dir (str): the directory to save the output events
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._last_write = -1
        self._group_rules = [
            (IsIn("/"), BaseRule()),
            (IsIn("loss"), Prefix("train")),
        ]

    def write(self):

        storage = get_event_storage()

        def _group_name(scalar_name):
            for (rule, op) in self._group_rules:
                if rule(scalar_name):
                    return op(scalar_name)
            return scalar_name

        stats = {
            _group_name(name): scalars[0]
            for name, scalars in storage.latest().items()
            if scalars[1] > self._last_write
        }
        if len(stats) > 0:
            self._last_write = max([v[1] for k, v in storage.latest().items()])

        # storage.put_{image,histogram} is only meant to be used by
        # tensorboard writer. So we access its internal fields directly from here.
        if len(storage._vis_data) >= 1:
            stats["image"] = [
                wandb.Image(img, caption=img_name)
                for img_name, img, step_num in storage._vis_data
            ]
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        if len(storage._histograms) >= 1:

            def create_bar(tag, bucket_limits, bucket_counts, **kwargs):
                data = [
                    [label, val] for (label, val) in zip(bucket_limits, bucket_counts)
                ]
                table = wandb.Table(data=data, columns=["label", "value"])
                return wandb.plot.bar(table, "label", "value", title=tag)

            stats["hist"] = [create_bar(**params) for params in storage._histograms]

            storage.clear_histograms()

        if len(stats) == 0:
            return
        wandb.log(stats, step=storage.iter)

    def close(self):
        wandb.finish()