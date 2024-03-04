from .cupy_kernels import remapping_kernel, patch_error_kernel, pairwise_patch_error_kernel
import numpy as np
import cupy as cp
import cv2


class PatchMatcher:
    def __init__(
        self, height, width, channel, minimum_patch_size,
        threads_per_block=8, num_iter=5, gpu_id=0, guide_weight=10.0,
        random_search_steps=3, random_search_range=4,
        use_mean_target_style=False, use_pairwise_patch_error=False,
        tracking_window_size=0
    ):
        self.height = height
        self.width = width
        self.channel = channel
        self.minimum_patch_size = minimum_patch_size
        self.threads_per_block = threads_per_block
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.guide_weight = guide_weight
        self.random_search_steps = random_search_steps
        self.random_search_range = random_search_range
        self.use_mean_target_style = use_mean_target_style
        self.use_pairwise_patch_error = use_pairwise_patch_error
        self.tracking_window_size = tracking_window_size

        self.patch_size_list = [minimum_patch_size + i*2 for i in range(num_iter)][::-1]
        self.pad_size = self.patch_size_list[0] // 2
        self.grid = (
            (height + threads_per_block - 1) // threads_per_block,
            (width + threads_per_block - 1) // threads_per_block
        )
        self.block = (threads_per_block, threads_per_block)

    def pad_image(self, image):
        return cp.pad(image, ((0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)))

    def unpad_image(self, image):
        return image[:, self.pad_size: -self.pad_size, self.pad_size: -self.pad_size, :]

    def apply_nnf_to_image(self, nnf, source):
        batch_size = source.shape[0]
        target = cp.zeros((batch_size, self.height + self.pad_size * 2, self.width + self.pad_size * 2, self.channel), dtype=cp.float32)
        remapping_kernel(
            self.grid + (batch_size,),
            self.block,
            (self.height, self.width, self.channel, self.patch_size, self.pad_size, source, nnf, target)
        )
        return target

    def get_patch_error(self, source, nnf, target):
        batch_size = source.shape[0]
        error = cp.zeros((batch_size, self.height, self.width), dtype=cp.float32)
        patch_error_kernel(
            self.grid + (batch_size,),
            self.block,
            (self.height, self.width, self.channel, self.patch_size, self.pad_size, source, nnf, target, error)
        )
        return error

    def get_pairwise_patch_error(self, source, nnf):
        batch_size = source.shape[0]//2
        error = cp.zeros((batch_size, self.height, self.width), dtype=cp.float32)
        source_a, nnf_a = source[0::2].copy(), nnf[0::2].copy()
        source_b, nnf_b = source[1::2].copy(), nnf[1::2].copy()
        pairwise_patch_error_kernel(
            self.grid + (batch_size,),
            self.block,
            (self.height, self.width, self.channel, self.patch_size, self.pad_size, source_a, nnf_a, source_b, nnf_b, error)
        )
        error = error.repeat(2, axis=0)
        return error

    def get_error(self, source_guide, target_guide, source_style, target_style, nnf):
        error_guide = self.get_patch_error(source_guide, nnf, target_guide)
        if self.use_mean_target_style:
            target_style = self.apply_nnf_to_image(nnf, source_style)
            target_style = target_style.mean(axis=0, keepdims=True)
            target_style = target_style.repeat(source_guide.shape[0], axis=0)
        if self.use_pairwise_patch_error:
            error_style = self.get_pairwise_patch_error(source_style, nnf)
        else:
            error_style = self.get_patch_error(source_style, nnf, target_style)
        error = error_guide * self.guide_weight + error_style
        return error

    def clamp_bound(self, nnf):
        nnf[:,:,:,0] = cp.clip(nnf[:,:,:,0], 0, self.height-1)
        nnf[:,:,:,1] = cp.clip(nnf[:,:,:,1], 0, self.width-1)
        return nnf

    def random_step(self, nnf, r):
        batch_size = nnf.shape[0]
        step = cp.random.randint(-r, r+1, size=(batch_size, self.height, self.width, 2), dtype=cp.int32)
        upd_nnf = self.clamp_bound(nnf + step)
        return upd_nnf

    def neighboor_step(self, nnf, d):
        if d==0:
            upd_nnf = cp.concatenate([nnf[:, :1, :], nnf[:, :-1, :]], axis=1)
            upd_nnf[:, :, :, 0] += 1
        elif d==1:
            upd_nnf = cp.concatenate([nnf[:, :, :1], nnf[:, :, :-1]], axis=2)
            upd_nnf[:, :, :, 1] += 1
        elif d==2:
            upd_nnf = cp.concatenate([nnf[:, 1:, :], nnf[:, -1:, :]], axis=1)
            upd_nnf[:, :, :, 0] -= 1
        elif d==3:
            upd_nnf = cp.concatenate([nnf[:, :, 1:], nnf[:, :, -1:]], axis=2)
            upd_nnf[:, :, :, 1] -= 1
        upd_nnf = self.clamp_bound(upd_nnf)
        return upd_nnf
        
    def shift_nnf(self, nnf, d):
        if d>0:
            d = min(nnf.shape[0], d)
            upd_nnf = cp.concatenate([nnf[d:]] + [nnf[-1:]] * d, axis=0)
        else:
            d = max(-nnf.shape[0], d)
            upd_nnf = cp.concatenate([nnf[:1]] * (-d) + [nnf[:d]], axis=0)
        return upd_nnf
    
    def track_step(self, nnf, d):
        if self.use_pairwise_patch_error:
            upd_nnf = cp.zeros_like(nnf)
            upd_nnf[0::2] = self.shift_nnf(nnf[0::2], d)
            upd_nnf[1::2] = self.shift_nnf(nnf[1::2], d)
        else:
            upd_nnf = self.shift_nnf(nnf, d)
        return upd_nnf

    def C(self, n, m):
        # not used
        c = 1
        for i in range(1, n+1):
            c *= i
        for i in range(1, m+1):
            c //= i
        for i in range(1, n-m+1):
            c //= i
        return c

    def bezier_step(self, nnf, r):
        # not used
        n = r * 2 - 1
        upd_nnf = cp.zeros(shape=nnf.shape, dtype=cp.float32)
        for i, d in enumerate(list(range(-r, 0)) + list(range(1, r+1))):
            if d>0:
                ctl_nnf = cp.concatenate([nnf[d:]] + [nnf[-1:]] * d, axis=0)
            elif d<0:
                ctl_nnf = cp.concatenate([nnf[:1]] * (-d) + [nnf[:d]], axis=0)
            upd_nnf += ctl_nnf * (self.C(n, i) / 2**n)
        upd_nnf = self.clamp_bound(upd_nnf).astype(nnf.dtype)
        return upd_nnf

    def update(self, source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf):
        upd_err = self.get_error(source_guide, target_guide, source_style, target_style, upd_nnf)
        upd_idx = (upd_err < err)
        nnf[upd_idx] = upd_nnf[upd_idx]
        err[upd_idx] = upd_err[upd_idx]
        return nnf, err

    def propagation(self, source_guide, target_guide, source_style, target_style, nnf, err):
        for d in cp.random.permutation(4):
            upd_nnf = self.neighboor_step(nnf, d)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
        return nnf, err
        
    def random_search(self, source_guide, target_guide, source_style, target_style, nnf, err):
        for i in range(self.random_search_steps):
            upd_nnf = self.random_step(nnf, self.random_search_range)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
        return nnf, err

    def track(self, source_guide, target_guide, source_style, target_style, nnf, err):
        for d in range(1, self.tracking_window_size + 1):
            upd_nnf = self.track_step(nnf, d)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
            upd_nnf = self.track_step(nnf, -d)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
        return nnf, err

    def iteration(self, source_guide, target_guide, source_style, target_style, nnf, err):
        nnf, err = self.propagation(source_guide, target_guide, source_style, target_style, nnf, err)
        nnf, err = self.random_search(source_guide, target_guide, source_style, target_style, nnf, err)
        nnf, err = self.track(source_guide, target_guide, source_style, target_style, nnf, err)
        return nnf, err

    def estimate_nnf(self, source_guide, target_guide, source_style, nnf):
        with cp.cuda.Device(self.gpu_id):
            source_guide = self.pad_image(source_guide)
            target_guide = self.pad_image(target_guide)
            source_style = self.pad_image(source_style)
            for it in range(self.num_iter):
                self.patch_size = self.patch_size_list[it]
                target_style = self.apply_nnf_to_image(nnf, source_style)
                err = self.get_error(source_guide, target_guide, source_style, target_style, nnf)
                nnf, err = self.iteration(source_guide, target_guide, source_style, target_style, nnf, err)
            target_style = self.unpad_image(self.apply_nnf_to_image(nnf, source_style))
        return nnf, target_style


class PyramidPatchMatcher:
    def __init__(
        self, image_height, image_width, channel, minimum_patch_size,
        threads_per_block=8, num_iter=5, gpu_id=0, guide_weight=10.0,
        use_mean_target_style=False, use_pairwise_patch_error=False,
        tracking_window_size=0,
        initialize="identity"
    ):
        maximum_patch_size = minimum_patch_size + (num_iter - 1) * 2
        self.pyramid_level = int(np.log2(min(image_height, image_width) / maximum_patch_size))
        self.pyramid_heights = []
        self.pyramid_widths = []
        self.patch_matchers = []
        self.minimum_patch_size = minimum_patch_size
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.initialize = initialize
        for level in range(self.pyramid_level):
            height = image_height//(2**(self.pyramid_level - 1 - level))
            width = image_width//(2**(self.pyramid_level - 1 - level))
            self.pyramid_heights.append(height)
            self.pyramid_widths.append(width)
            self.patch_matchers.append(PatchMatcher(
                height, width, channel, minimum_patch_size=minimum_patch_size,
                threads_per_block=threads_per_block, num_iter=num_iter, gpu_id=gpu_id, guide_weight=guide_weight,
                use_mean_target_style=use_mean_target_style, use_pairwise_patch_error=use_pairwise_patch_error,
                tracking_window_size=tracking_window_size
            ))

    def resample_image(self, images, level):
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        images = images.get()
        images_resample = []
        for image in images:
            image_resample = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            images_resample.append(image_resample)
        images_resample = cp.array(np.stack(images_resample), dtype=cp.float32)
        return images_resample

    def initialize_nnf(self, batch_size):
        if self.initialize == "random":
            height, width = self.pyramid_heights[0], self.pyramid_widths[0]
            nnf = cp.stack([
                cp.random.randint(0, height, (batch_size, height, width), dtype=cp.int32),
                cp.random.randint(0, width, (batch_size, height, width), dtype=cp.int32)
            ], axis=3)
        elif self.initialize == "identity":
            height, width = self.pyramid_heights[0], self.pyramid_widths[0]
            nnf = cp.stack([
                cp.repeat(cp.arange(height), width).reshape(height, width),
                cp.tile(cp.arange(width), height).reshape(height, width)
            ], axis=2)
            nnf = cp.stack([nnf] * batch_size)
        else:
            raise NotImplementedError()
        return nnf

    def update_nnf(self, nnf, level):
        # upscale
        nnf = nnf.repeat(2, axis=1).repeat(2, axis=2) * 2
        nnf[:,[i for i in range(nnf.shape[0]) if i&1],:,0] += 1
        nnf[:,:,[i for i in range(nnf.shape[0]) if i&1],1] += 1
        # check if scale is 2
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        if height != nnf.shape[0] * 2 or width != nnf.shape[1] * 2:
            nnf = nnf.get().astype(np.float32)
            nnf = [cv2.resize(n, (width, height), interpolation=cv2.INTER_LINEAR) for n in nnf]
            nnf = cp.array(np.stack(nnf), dtype=cp.int32)
            nnf = self.patch_matchers[level].clamp_bound(nnf)
        return nnf

    def apply_nnf_to_image(self, nnf, image):
        with cp.cuda.Device(self.gpu_id):
            image = self.patch_matchers[-1].pad_image(image)
            image = self.patch_matchers[-1].apply_nnf_to_image(nnf, image)
        return image

    def estimate_nnf(self, source_guide, target_guide, source_style):
        with cp.cuda.Device(self.gpu_id):
            if not isinstance(source_guide, cp.ndarray):
                source_guide = cp.array(source_guide, dtype=cp.float32)
            if not isinstance(target_guide, cp.ndarray):
                target_guide = cp.array(target_guide, dtype=cp.float32)
            if not isinstance(source_style, cp.ndarray):
                source_style = cp.array(source_style, dtype=cp.float32)
            for level in range(self.pyramid_level):
                nnf = self.initialize_nnf(source_guide.shape[0]) if level==0 else self.update_nnf(nnf, level)
                source_guide_ = self.resample_image(source_guide, level)
                target_guide_ = self.resample_image(target_guide, level)
                source_style_ = self.resample_image(source_style, level)
                nnf, target_style = self.patch_matchers[level].estimate_nnf(
                    source_guide_, target_guide_, source_style_, nnf
                )
        return nnf.get(), target_style.get()
