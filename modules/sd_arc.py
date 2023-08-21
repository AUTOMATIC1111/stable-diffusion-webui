import collections
from modules import devices
import time
import os
import gc
import torch
from modules.shared import cmd_opts
from modules import shared
from modules import paths
import pickle
import logging
import psutil


"""
use in two places via a SpecifiedCache model.
1 sd_models.py
2 processing.py
"""
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
prefix = "arc"

def pickle_load(filepath):
    # Define a custom find_global function
    def find_global(module, name):
        # Return a safe replacement object
        if name == 'MyClass':
            return None
        # Return the default lookup function
        return getattr(module, name)

    # Open the file
    with open(filepath, 'rb') as f:
        # Create an Unpickler object and set the find_global function
        unpickler = pickle.Unpickler(f)
        unpickler.find_global = find_global
        # Load the Python object from the file
        my_obj = unpickler.load()
        return my_obj


def get_memory():
    try:
        process = psutil.Process(os.getpid())
        # only rss is cross-platform guaranteed so we dont rely on other values
        res = process.memory_info()
        # and total memory is calculated as actual value is not cross-platform safe
        ram_total = 100 * res.rss / process.memory_percent()
        ram = {'free': ram_total - res.rss,
               'used': res.rss, 'total': ram_total}
    except Exception as err:
        ram = {'error': f'{err}'}
    try:
        if torch.cuda.is_available():
            s = torch.cuda.mem_get_info()
            system = {'free': s[0], 'used': s[1] - s[0], 'total': s[1]}
            s = dict(torch.cuda.memory_stats(shared.device))
            allocated = {
                'current': s['allocated_bytes.all.current'], 'peak': s['allocated_bytes.all.peak']}
            reserved = {
                'current': s['reserved_bytes.all.current'], 'peak': s['reserved_bytes.all.peak']}
            active = {'current': s['active_bytes.all.current'],
                      'peak': s['active_bytes.all.peak']}
            inactive = {'current': s['inactive_split_bytes.all.current'],
                        'peak': s['inactive_split_bytes.all.peak']}
            warnings = {
                'retries': s['num_alloc_retries'], 'oom': s['num_ooms']}
            cuda = {
                'system': system,
                'active': active,
                'allocated': allocated,
                'reserved': reserved,
                'inactive': inactive,
                'events': warnings,
            }
        else:
            cuda = {'error': 'unavailable'}
    except Exception as err:
        cuda = {'error': f'{err}'}
    return dict(ram=ram, cuda=cuda)


class SpecifiedCache:
    def __init__(self) -> None:
        self.model_size = 5.5 if cmd_opts.no_half else (2.56 if cmd_opts.no_half_vae else 2.39)
        self.model_size_xl = 13 if cmd_opts.no_half else (9 if cmd_opts.no_half_vae else 7)
        self.model_size_disk = 2.2
        self.size_base = 2.5 if cmd_opts.no_half or cmd_opts.no_half_vae else 0.5
        self.size_base_xl = 3. if cmd_opts.no_half or cmd_opts.no_half_vae else 1
        self.batch_base = 0.3
        self.batch_base_xl = 0.5
        self.ram_model_size = 5
        self.cuda_model_ram = 3

        self.cuda_keep_size = 2
        self.ram_keep_size = 5
        self.disk_keep_size = 0

        self.gpu_specified_models = None
        self.ram_specified_models = None
        self.reload_time = {}
        self.reload_count = {}
        self.checkpoint_file = {}

        self.lru = collections.OrderedDict()
        self.ram = collections.OrderedDict()
        self.disk = collections.OrderedDict()

        gpu_memory_size = self.get_free_cuda()
        ram_size = self.get_free_ram()
        disk_size = self.get_free_disk()
        if shared.cmd_opts.arc:
            logging.info(">>>>> arc: multiple-level cache enabled <<<<<.")
            logging.info(f"{prefix}: gpu memory：{gpu_memory_size : .0f} GB, ram:{ram_size : .0f} GB，disk:{disk_size : .0f} GB")

    def get_free_cuda(self):
        sysinfo = get_memory()
        free_size = sysinfo.get('cuda', {}).get('system', {}).get('free', 24*1024**3)/1024**3
        return free_size

    def get_free_ram(self):
        sysinfo = get_memory()
        used_size = sysinfo.get('ram', {}).get('used', 32*1024**3)/1024**3
        if shared.cmd_opts.system_ram_size:
            specified_free_size = shared.cmd_opts.system_ram_size - used_size
            logging.debug(f"free size method 1: {specified_free_size}")
            return shared.cmd_opts.system_ram_size - used_size
        if shared.opts.sd_checkpoint_cache is not None:
            rectified_free_size = shared.opts.sd_checkpoint_cache * 5 - \
                len(self.lru) * self.cuda_model_ram - len(self.ram) * 5
            logging.debug(f"free size method 2: {rectified_free_size}")
            return rectified_free_size
        free_size = 0
        logging.debug(f"free size method 3: {free_size}")
        return free_size

    def get_system_free_ram(self):
        sysinfo = get_memory()
        free_size = sysinfo.get('ram', {}).get('free', 32*1024**3)/1024**3
        return free_size

    def get_free_disk(self):
        if shared.cmd_opts.arc_disk_size:
            free_space_gb = shared.cmd_opts.arc_disk_size - len(self.disk) * self.model_size_disk
            return free_space_gb
        return 0

    def set_specified(self, gpu_filenames: list, ram_filenames: list):
        not_exist = []
        for filename in gpu_filenames + ram_filenames:
            if not os.path.exists(os.path.join(model_path, filename)):
                not_exist.append(filename)
        self.gpu_specified_models = gpu_filenames
        self.ram_specified_models = ram_filenames
        return not_exist

    def is_gpu_specified(self, key):
        return self.gpu_specified_models is None or self.get_model_name(key) in self.gpu_specified_models

    def is_ram_specified(self, key):
        return self.ram_specified_models is None or self.get_model_name(key) in self.ram_specified_models

    def lru_pop(self, key):
        logging.info('using model cached in device')
        value = self.lru.pop(key)
        if self.is_cuda(value):
            return value
        return None

    def ram_pop(self, key):
        logging.info('using model cached in ram')
        value = self.ram.pop(key)
        if not self.is_cuda(value):
            return value.to(devices.device)
        return None

    def disk_pop(self, key):
        try:
            logging.info('using model cached in disk')
            value = self.pickle_load(key)
            if not self.is_cuda(value):
                return value.to(devices.device)
        except Exception as e:
            logging.info(str(e))
        return None

    def pop(self, key):
        self.reload(key)
        if self.lru.get(key) is not None:
            return self.lru_pop(key)
        if self.ram.get(key) is not None:
            return self.ram_pop(key)
        if self.disk.get(key) is not None:
            return self.disk_pop(key)
        return None

    def is_cuda(self, value):
        return 'cuda' in str(value.device)

    def contains(self, key):
        return (key in self.lru) or (key in self.ram) or (key in self.disk)

    def delete_oldest(self):
        cudas = [k for k, v in self.lru.items()]
        if len(cudas) == 0:
            return False
        sorted_cudas = sorted(cudas, key=lambda x: self.reload_time.get(x, 0))
        oldest = sorted_cudas[0]
        del sorted_cudas
        del cudas
        logging.info(f"delete cache: {oldest}")
        v = self.lru.pop(oldest)
        res = self.put_ram(oldest, v)
        if not res:
            v.to(device="meta")
        del oldest
        del v
        return True

    def get_model_size(self, config):
        model_size = self.model_size_xl
        if config:
            config_name = os.path.basename(config)
            logging.info(config_name)
            # ["sd-xl-refiner.yaml", "sd_xl_refiner.yaml"]:
            if '-xl-' in config_name or '_xl_' in config_name:
                model_size = self.model_size_xl
            # ["v2-inference-v.yaml", "v1-inference.yaml", "v1-inpainting-inference.yaml"]
            elif 'v1' in config_name or 'v2' in config_name:
                model_size = self.model_size
            else:
                logging.error(f"unkown config: {config_name}")
        return model_size

    def prepare_memory(self, config):
        """
        prepare cuda and ram memory for model.
        """
        model_size = self.get_model_size(config)
        is_delete = False
        while (self.get_free_cuda() < model_size + self.cuda_keep_size or self.get_system_free_ram() < self.ram_keep_size) and len(self.lru) > 0:
            tmp = self.delete_oldest()
            is_delete = is_delete or tmp
        if is_delete:
            self.cuda_gc()

    def put_lru(self, key, value):
        """
        value must be cuda.
        """
        self.prepare_memory(value.used_config)
        logging.info(f"add cache: {key}")
        self.lru[key] = value
        if self.ram.get(key) is not None:
            v = self.ram.pop(key)
            logging.info(f"checkpoint should not exists: {key}")
            del v
            gc.collect()

    def get_model_name(self, key):
        return os.path.basename(key)

    def put(self, key, value):
        if not self.is_cuda(value):
            logging.info(f"not cache, not in cuda: {key}")
            return

        if self.contains(key):
            return

        if self.is_gpu_specified(key):
            self.put_lru(key, value)
            return
        logging.info(f"not cache: {key}")
        del value

    def reload(self, key):
        self.reload_time[key] = time.time_ns()
        if key not in self.reload_count:
            self.reload_count[key] = 0
        self.reload_count[key] += 1

    def release_memory(self, p):
        """
        prepare cuda memory for controlnet, height*width*batch_size
        """
        try:
            start_time = time.time()
            size_base = self.size_base_xl
            if p.sd_model.is_sd1 or p.sd_model.is_sd2:
                size_base = self.size_base

            need_size = (p.height * p.width / (512*512) - 1) * (size_base + self.batch_base) + 4 # 4 is keep size
            for item in p.script_args:
                if ("controlnet" in str(type(item)).lower() and item.enabled) or (item.isinstance(dict) and item.get("model") is not None):
                    need_size += 0.7
                    logging.info("prepare memory for controlnet")
            is_delete = False
            while self.get_free_cuda() < need_size and len(self.lru) > 0:
                tmp = self.delete_oldest()
                is_delete = is_delete or tmp
            if is_delete:
                self.cuda_gc()
            logging.info(f"prepare memory: {need_size:.2f} GB, free memory after release: { self.get_free_cuda():.2f}GB, time cost: {time.time() - start_time:.1f} s")
        except Exception as e:
            raise e

    def cuda_gc(self):
        start_time = time.time()
        gc.collect()
        devices.torch_gc()
        torch.cuda.empty_cache()
        logging.info(f"cuda garbage collect cost: {time.time()-start_time:.1f} s")

    def ram_gc(self):
        start_time = time.time()
        gc.collect()
        devices.torch_gc()
        logging.info(
            f"ram garbage collect cost: {time.time()-start_time:.1f} s")

    def put_ram(self, key, value):
        if self.is_ram_specified(key):
            is_delete = False
            while self.get_free_ram() < self.ram_model_size and len(self.ram) > 0:
                tmp = self.delete_ram()
                is_delete = is_delete or tmp
            if is_delete:
                self.ram_gc()
            if self.get_free_ram() > self.ram_model_size:
                if self.is_cuda(value):
                    value.to(devices.cpu)
                self.ram[key] = value
                logging.info(f"add ram cache: {key}")
                return True
        del value
        del key
        return False

    def is_disk_specified(self, key):
        return self.is_ram_specified(key) or self.is_gpu_specified(key)

    def pickle_name(self, key):
        if not os.path.exists(os.path.join(model_path, 'pkl')):
            os.mkdir(os.path.join(model_path, 'pkl'))
        return os.path.join(model_path, 'pkl', os.path.basename(key)+'.pkl')

    def pickle_dump(self, key, value):
        start_time = time.time()
        pickle_path = self.pickle_name(key)
        if not os.path.exists(pickle_path):
            pickle.dump(value, open(pickle_path, 'wb'))
        logging.info(f"save disk cost: {time.time()-start_time:.2f} s")
        return pickle_path

    def pickle_load(self, key):
        tmp = shared.cmd_args.disable_safe_unpickle
        try:
            shared.cmd_args.disable_safe_unpickle = True
            start_time = time.time()
            pickle_path = self.pickle_name(key)
            value = pickle_load(pickle_path)
            logging.info(f"read disk cost: {time.time()-start_time:.2f} s")
        finally:
            shared.cmd_args.disable_safe_unpickle = tmp
        return value

    def delete_disk(self):
        if len(self.disk) == 0:
            return
        ckpts = self.disk.keys().list()
        sorted_disks = sorted(ckpts, key=lambda x: self.reload_count.get(x, 0))
        least = sorted_disks[0]
        del ckpts
        del sorted_disks
        logging.info(f"delete disk cache: {least}")
        v = self.disk.pop(least)
        os.remove(v)
        del least

    def is_more_disk_frequent(self, key):
        if len(self.disk) == 0:
            return True
        ckpts = self.disk.keys().list()
        sorted_disks = sorted(ckpts, key=lambda x: self.reload_count.get(x, 0))
        least = sorted_disks[0]
        if self.reload_count.get(key, 0) > self.reload_count.get(least, 0):
            return True
        return False

    def put_disk(self, key, value):
        while self.get_free_disk() - self.disk_keep_size < self.model_size_disk and len(self.disk) > 0:
            self.delete_disk()

        if not (self.get_free_disk() - self.disk_keep_size > self.model_size_disk and self.is_more_disk_frequent(key)):
            del value
            del key
            return

        if self.is_disk_specified(key):
            if self.is_cuda(value):
                value.to(devices.cpu)
            self.disk[key] = self.pickle_dump(key, value)
            logging.info(f"add disk cache: {key}")
            return
        del value
        del key

    def delete_ram(self,):
        if len(self.ram) == 0:
            return False
        ckpts = self.ram.keys().list()
        sorted_rams = sorted(ckpts, key=lambda x: self.reload_time.get(x, 0))
        oldest = sorted_rams[0]
        del ckpts
        del sorted_rams
        logging.info(f"delete ram cache: {oldest}")
        v = self.ram.pop(oldest)
        if shared.cmd_opts.arc_disk_size:
            self.put_disk(oldest, v)
        del v
        del oldest
        return True

    def get_cudas(self):
        return [self.get_model_name(i) for i in self.lru.keys()]

    def get_rams(self):
        return [self.get_model_name(i) for i in self.ram.keys()]
