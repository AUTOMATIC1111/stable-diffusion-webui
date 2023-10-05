from ray import serve
import ray
from fastapi import FastAPI
from modules.api.raypi import Raypi
from modules import initialize_util
from modules import script_callbacks
from modules import initialize
import time

from ray.serve.handle import DeploymentHandle
from modules.call_queue import queue_lock
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

initialize.initialize()
app = FastAPI()
#app.include_router(Raypi(app).router)
initialize_util.setup_middleware(app)
script_callbacks.before_ui_callback()
script_callbacks.app_started_callback(None, app)







def ray_only():
    serve.shutdown()
    serve.start()
    serve.run(Raypi.bind(), port=8000)  #route_prefix="/sdapi/v1" # Call the launch_ray method to get the FastAPI app


    print("Done setting up replicas! Now accepting requests...")
    while True:
        time.sleep(1000)
