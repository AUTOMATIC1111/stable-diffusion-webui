from ray import serve
import ray

from modules.api.raypi import Raypi

import time
import os


ray.init()
#ray.init("ray://localhost:10001")


def ray_only():
    serve.shutdown()
    #serve.start(port=int(os.environ.get("RAY_PORT", 8000)))
    serve.start(detached=True, http_options={
                                            "host": os.environ.get("RAY_IP", "0.0.0.0"), 
                                            "port": int(os.environ.get("RAY_PORT", 8000))
                                            }
                )
    print(f"Starting Raypi on port {os.environ.get('RAY_PORT', 8000)}")
    serve.run(Raypi.bind(), port=int(os.environ.get("RAY_PORT", 8000)), route_prefix="/sdapi/v1")  #route_prefix="/sdapi/v1" # Call the launch_ray method to get the FastAPI app
    print("Done setting up replicas! Now accepting requests...")
    while True:
        time.sleep(1000)
