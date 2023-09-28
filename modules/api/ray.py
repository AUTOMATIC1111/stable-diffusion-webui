from ray import serve
import ray
from fastapi import FastAPI
from modules.api.api import Api
from modules.call_queue import queue_lock
from modules import initialize_util



#ray.init()
ray.init("ray://localhost:10001")

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
api = Api(app)
# Try to serialize the Api class
#pickle.dumps(api)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    num_replicas=NUM_REPLICAS,
)
@serve.ingress(api)
class APIIngress(Api):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



