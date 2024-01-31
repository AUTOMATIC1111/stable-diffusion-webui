# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import pickle
from urllib.parse import parse_qs, urlparse
import torch
from fvcore.common.checkpoint import Checkpointer
from torch.nn.parallel import DistributedDataParallel

import annotator.oneformer.detectron2.utils.comm as comm
from annotator.oneformer.detectron2.utils.file_io import PathManager

from .c2_model_loading import align_and_update_state_dicts


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to:
    1. handle models in detectron & detectron2 model zoo, and apply conversions for legacy models.
    2. correctly load checkpoints that are only available on the master worker
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.path_manager = PathManager
        self._parsed_url_during_load = None

    def load(self, path, *args, **kwargs):
        assert self._parsed_url_during_load is None
        need_sync = False
        logger = logging.getLogger(__name__)
        logger.info("[DetectionCheckpointer] Loading from {} ...".format(path))

        if path and isinstance(self.model, DistributedDataParallel):
            path = self.path_manager.get_local_path(path)
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                logger.warning(
                    f"Not all workers can read checkpoint {path}. "
                    "Training may fail to fully resume."
                )
                # TODO: broadcast the checkpoint file contents from main
                # worker, and load from it instead.
                need_sync = True
            if not has_file:
                path = None  # don't load if not readable

        if path:
            parsed_url = urlparse(path)
            self._parsed_url_during_load = parsed_url
            path = parsed_url._replace(query="").geturl()  # remove query from filename
            path = self.path_manager.get_local_path(path)

        self.logger.setLevel('CRITICAL')
        ret = super().load(path, *args, **kwargs)

        if need_sync:
            logger.info("Broadcasting model states from main worker ...")
            self.model._sync_params_and_buffers()
        self._parsed_url_during_load = None  # reset to None
        return ret

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
        elif filename.endswith(".pyth"):
            # assume file is from pycls; no one else seems to use the ".pyth" extension
            with PathManager.open(filename, "rb") as f:
                data = torch.load(f)
            assert (
                "model_state" in data
            ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
            model_state = {
                k: v
                for k, v in data["model_state"].items()
                if not k.endswith("num_batches_tracked")
            }
            return {"model": model_state, "__author__": "pycls", "matching_heuristics": True}

        loaded = self._torch_load(filename)
        if "model" not in loaded:
            loaded = {"model": loaded}
        assert self._parsed_url_during_load is not None, "`_load_file` must be called inside `load`"
        parsed_url = self._parsed_url_during_load
        queries = parse_qs(parsed_url.query)
        if queries.pop("matching_heuristics", "False") == ["True"]:
            loaded["matching_heuristics"] = True
        if len(queries) > 0:
            raise ValueError(
                f"Unsupported query remaining: f{queries}, orginal filename: {parsed_url.geturl()}"
            )
        return loaded

    def _torch_load(self, f):
        return super()._load_file(f)

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            checkpoint["model"] = align_and_update_state_dicts(
                self.model.state_dict(),
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        for k in incompatible.unexpected_keys[:]:
            # Ignore unexpected keys about cell anchors. They exist in old checkpoints
            # but now they are non-persistent buffers and will not be in new checkpoints.
            if "anchor_generator.cell_anchors" in k:
                incompatible.unexpected_keys.remove(k)
        return incompatible
