from modules import shared, sd_hijack, devices
from modules.api import models
from modules.textual_inversion.preprocess import preprocess


def post_create_embedding(args: dict):
    from modules.textual_inversion.textual_inversion import create_embedding
    try:
        shared.state.begin('api-embedding')
        filename = create_embedding(**args) # create empty embedding
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings() # reload embeddings so new one can be immediately used
        shared.state.end()
        return models.CreateResponse(info = f"create embedding filename: {filename}")
    except AssertionError as e:
        shared.state.end()
        return models.TrainResponse(info = f"create embedding error: {e}")

def post_create_hypernetwork(args: dict):
    from modules.hypernetworks.hypernetwork import create_hypernetwork
    try:
        shared.state.begin('api-hypernetwork')
        filename = create_hypernetwork(**args) # create empty embedding # pylint: disable=E1111
        shared.state.end()
        return models.CreateResponse(info = f"create hypernetwork filename: {filename}")
    except AssertionError as e:
        shared.state.end()
        return models.TrainResponse(info = f"create hypernetwork error: {e}")

def post_preprocess(args: dict):
    try:
        shared.state.begin('api-preprocess')
        preprocess(**args) # quick operation unless blip/booru interrogation is enabled
        shared.state.end()
        return models.PreprocessResponse(info = 'preprocess complete')
    except KeyError as e:
        shared.state.end()
        return models.PreprocessResponse(info = f"preprocess error: invalid token: {e}")
    except AssertionError as e:
        shared.state.end()
        return models.PreprocessResponse(info = f"preprocess error: {e}")
    except FileNotFoundError as e:
        shared.state.end()
        return models.PreprocessResponse(info = f'preprocess error: {e}')

def post_train_embedding(args: dict):
    from modules.textual_inversion.textual_inversion import train_embedding
    try:
        shared.state.begin('api-embedding')
        apply_optimizations = False
        error = None
        filename = ''
        if not apply_optimizations:
            sd_hijack.undo_optimizations()
        try:
            _embedding, filename = train_embedding(**args) # can take a long time to complete
        except Exception as e:
            error = e
        finally:
            if not apply_optimizations:
                sd_hijack.apply_optimizations()
            shared.state.end()
        return models.TrainResponse(info = f"train embedding complete: filename: {filename} error: {error}")
    except AssertionError as msg:
        shared.state.end()
        return models.TrainResponse(info = f"train embedding error: {msg}")

def post_train_hypernetwork(args: dict):
    from modules.hypernetworks.hypernetwork import train_hypernetwork
    try:
        shared.state.begin('api-hypernetwork')
        shared.loaded_hypernetworks = []
        apply_optimizations = False
        error = None
        filename = ''
        if not apply_optimizations:
            sd_hijack.undo_optimizations()
        try:
            _hypernetwork, filename = train_hypernetwork(**args)
        except Exception as e:
            error = e
        finally:
            shared.sd_model.cond_stage_model.to(devices.device)
            shared.sd_model.first_stage_model.to(devices.device)
            if not apply_optimizations:
                sd_hijack.apply_optimizations()
            shared.state.end()
        return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
    except AssertionError:
        shared.state.end()
        return models.TrainResponse(info=f"train embedding error: {error}")
