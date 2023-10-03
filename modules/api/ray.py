from ray import serve
import ray
from fastapi import FastAPI
from modules.api.api import Api

from modules import initialize_util
from modules import script_callbacks
from modules import initialize
import time


ray.init()
#ray.init("ray://localhost:10001")



NUM_REPLICAS: int = 1
if NUM_REPLICAS > ray.available_resources()["GPU"]:
    print(
        "Your cluster does not currently have enough resources to run with these settings. "
        "Consider decreasing the number of workers, or decreasing the resources needed "
        "per worker. Ignore this if your cluster auto-scales."
    )

app = FastAPI()
initialize_util.setup_middleware(app)
#api = Api(app)
script_callbacks.before_ui_callback()
script_callbacks.app_started_callback(None, app)

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    num_replicas=NUM_REPLICAS,
)
@serve.ingress(app)
class APIIngress:
    def __init__(self, *args, **kwargs) -> None:
        pass


def ray_only():
    # Shutdown any existing Serve replicas, if they're still around.
    serve.shutdown()
    serve.run(APIIngress.bind(), port=8000, name="serving_stable_diffusion_template")
    print("Done setting up replicas! Now accepting requests...")
    while True:
        time.sleep(1000)


