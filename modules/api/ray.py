from ray import serve
import ray

from modules.api.raypi import Raypi

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



def ray_only():
    serve.shutdown()
    serve.start()
    #Raypi.deploy()

    
    serve.run(Raypi.bind(), port=8000, route_prefix="/sdapi/v1")  #route_prefix="/sdapi/v1" # Call the launch_ray method to get the FastAPI app


    print("Done setting up replicas! Now accepting requests...")
    while True:
        time.sleep(1000)
