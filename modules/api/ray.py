from ray import serve
import ray

from modules.api.raypi import Raypi

import time
import os


#ray.init(os.environ.get("RAY_HEAD_ADDRESS", ""))
#ray.init("ray://localhost:10001")


#ray_head_address = os.environ.get("RAY_ADDRESS")
#print("RAY_ADDRESS:", ray_head_address)
#
#if ray_head_address:
#    #ray.init(address=ray_head_address)
#    ray.init(address=ray_head_address)
#    #ray.init(address="172.21.0.3:6388")
#else:
#    ray.init()
#

entrypoint = Raypi.bind()

def ray_only():
    serve.shutdown()
    if "RAY_DOCKER" in os.environ:
        print("starting ray in docker")
        serve.start(
            detached=True,
            http_options={
                        "host": os.environ.get("RAY_IP", "0.0.0.0"), 
                        "port": int(os.environ.get("RAY_PORT", 8000))
                        }
        )
    else:
        serve.start(
            http_options={
                        "host": os.environ.get("RAY_IP", "0.0.0.0"), 
                        "port": int(os.environ.get("RAY_PORT", 8000))
                        }
        )
    print(f"Starting Raypi on port {os.environ.get('RAY_PORT', 8000)}")
    serve.run(Raypi.bind(), port=int(os.environ.get("RAY_PORT", 8000)), route_prefix="/sdapi/v1")  #route_prefix="/sdapi/v1" # Call the launch_ray method to get the FastAPI app
    print("Done setting up replicas! Now accepting requests...")
    while True:
        time.sleep(1000)