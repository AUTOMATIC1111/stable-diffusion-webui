from ray import serve
import ray
from fastapi import FastAPI
from modules.api.raypi import Raypi
from modules.api.api import Raypi
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

@serve.deployment(    
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
    #route_prefix="/sdapi/v1",
    )
@serve.ingress(app)
class RayDeployment:
    def __init__(self):
        pass


# 2: Deploy the deployment.




def ray_only():
    serve.shutdown()
    serve.start()
    #Raypi.deploy()

    
    serve.run(Raypi.bind(), port=8000, route_prefix="/sdapi/v1")  #route_prefix="/sdapi/v1" # Call the launch_ray method to get the FastAPI app


    print("Done setting up replicas! Now accepting requests...")
    while True:
        time.sleep(1000)
