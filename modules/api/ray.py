from ray import serve
import ray
from fastapi import FastAPI
from modules.api.api import Api
from modules import initialize_util
from modules import script_callbacks
from modules import initialize
import time


from modules.shared_cmd_options import cmd_opts

ray.init()
#ray.init("ray://localhost:10001")



NUM_REPLICAS: int = 1
if NUM_REPLICAS > ray.available_resources()["GPU"]:
    print(
        "Your cluster does not currently have enough resources to run with these settings. "
        "Consider decreasing the number of workers, or decreasing the resources needed "
        "per worker. Ignore this if your cluster auto-scales."
    )

#initialize.initialize()
app = FastAPI()
#initialize_util.setup_middleware(app)
#api = Api(app)
#app.include_router(api.router)
#script_callbacks.before_ui_callback()
#script_callbacks.app_started_callback(None, app)



@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    num_replicas=NUM_REPLICAS,
)
@serve.ingress(app)
class APIIngress:
    def __init__(self) -> None:
        pass
        


def ray_only():
    from fastapi import FastAPI
    from modules.shared_cmd_options import cmd_opts

    initialize.initialize()

    app = FastAPI()
    initialize_util.setup_middleware(app)
    api = Api(app)
    app.include_router(api.router)

    from modules import script_callbacks
    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)

    # Shutdown any existing Serve replicas, if they're still around.
    serve.shutdown()
    serve.run(APIIngress.bind(), port=8000, name="serving_stable_diffusion_template")
    print("Done setting up replicas! Now accepting requests...")
    while True:
        time.sleep(1000)