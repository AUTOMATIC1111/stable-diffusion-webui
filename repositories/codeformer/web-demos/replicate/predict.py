"""
This file is used for deploying replicate demo:
https://replicate.com/sczhou/codeformer
running: cog predict -i image=@inputs/whole_imgs/04.jpg -i codeformer_fidelity=0.5 -i upscale=2
push: cog push r8.im/sczhou/codeformer
"""

import tempfile
import cv2
import torch
from torchvision.transforms.functional import normalize
try:
    from cog import BasePredictor, Input, Path
except Exception:
    print('please install cog package')

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda:0"
        self.upsampler = set_realesrgan()
        self.net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        ckpt_path = "weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path)[
            "params_ema"
        ]  # update file permission if cannot load
        self.net.load_state_dict(checkpoint)
        self.net.eval()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        codeformer_fidelity: float = Input(
            default=0.5,
            ge=0,
            le=1,
            description="Balance the quality (lower number) and fidelity (higher number).",
        ),
        background_enhance: bool = Input(
            description="Enhance background image with Real-ESRGAN", default=True
        ),
        face_upsample: bool = Input(
            description="Upsample restored faces for high-resolution AI-created images",
            default=True,
        ),
        upscale: int = Input(
            description="The final upsampling scale of the image",
            default=2,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # take the default setting for the demo
        has_aligned = False
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=self.device,
        )

        bg_upsampler = self.upsampler if background_enhance else None
        face_upsampler = self.upsampler if face_upsample else None

        img = cv2.imread(str(image), cv2.IMREAD_COLOR)

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            print(f"\tdetect {num_det_faces} faces")
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f"\tFailed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )

        # save restored img
        out_path = Path(tempfile.mkdtemp()) / 'output.png'
        imwrite(restored_img, str(out_path))

        return out_path


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def set_realesrgan():
    if not torch.cuda.is_available():  # CPU
        import warnings

        warnings.warn(
            "The unoptimized RealESRGAN is slow on CPU. We do not use it. "
            "If you really want to use it, please modify the corresponding codes.",
            category=RuntimeWarning,
        )
        upsampler = None
    else:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path="./weights/realesrgan/RealESRGAN_x2plus.pth",
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=True,
        )
    return upsampler
