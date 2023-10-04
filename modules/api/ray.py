from ray import serve
import ray
from fastapi import FastAPI
from modules.api.raypi import Api
from modules import initialize_util
from modules import script_callbacks
from modules import initialize
import time

from ray.serve.handle import DeploymentHandle

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
#app = FastAPI()
#initialize_util.setup_middleware(app)
#api = Api(app)
#app.include_router(api.router)
#script_callbacks.before_ui_callback()
#script_callbacks.app_started_callback(None, app)


def ray_only():
    from fastapi import FastAPI
    from modules.shared_cmd_options import cmd_opts
    from modules import script_callbacks
    # Shutdown any existing Serve replicas, if they're still around.
    serve.shutdown()
    serve.start()

    initialize.initialize()

    app = FastAPI()
    initialize_util.setup_middleware(app)

    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)
    #Api.deploy()
    #api = Api(app)  # Create an instance of the Api class
    serve.run(Api.bind() , port=8000)  #route_prefix="/sdapi/v1" # Call the launch_ray method to get the FastAPI app


    print("Done setting up replicas! Now accepting requests...")
    while True:
        time.sleep(1000)



