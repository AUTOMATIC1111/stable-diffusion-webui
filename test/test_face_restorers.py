import os
from test.conftest import test_files_path, test_outputs_path

import numpy as np
import pytest
from PIL import Image


@pytest.mark.usefixtures("initialize")
@pytest.mark.parametrize("restorer_name", ["gfpgan", "codeformer"])
def test_face_restorers(restorer_name):
    from modules import shared

    if restorer_name == "gfpgan":
        from modules import gfpgan_model
        gfpgan_model.setup_model(shared.cmd_opts.gfpgan_models_path)
        restorer = gfpgan_model.gfpgan_fix_faces
    elif restorer_name == "codeformer":
        from modules import codeformer_model
        codeformer_model.setup_model(shared.cmd_opts.codeformer_models_path)
        restorer = codeformer_model.codeformer.restore
    else:
        raise NotImplementedError("...")
    img = Image.open(os.path.join(test_files_path, "two-faces.jpg"))
    np_img = np.array(img, dtype=np.uint8)
    fixed_image = restorer(np_img)
    assert fixed_image.shape == np_img.shape
    assert not np.allclose(fixed_image, np_img)  # should have visibly changed
    Image.fromarray(fixed_image).save(os.path.join(test_outputs_path, f"{restorer_name}.png"))
