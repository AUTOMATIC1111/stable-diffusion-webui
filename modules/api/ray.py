from ray import serve
import ray
from fastapi import FastAPI
from modules.api.api import Api
from modules.call_queue import queue_lock
from modules import initialize_util
from modules import script_callbacks
from modules import initialize



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

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    num_replicas=NUM_REPLICAS,
)
@serve.ingress(app)
class APIIngress(Api):
    def __init__(self, *args, **kwargs) -> None:
        from launch import ray_launch
        #from modules import sd_samplers
        ray_launch()
        #sd_samplers.set_samplers()
        initialize.imports()
        initialize.check_versions()

        initialize.initialize()
        app = FastAPI()
        initialize_util.setup_middleware(app)


        api = Api(app)

        script_callbacks.before_ui_callback()
        script_callbacks.app_started_callback(None, app)

        print(f"Startup time: {startup_timer.summary()}.")
        api.launch(
            server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1",
            port=cmd_opts.port if cmd_opts.port else 7861,
            root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else ""
        )

        super().__init__(*args, **kwargs)



