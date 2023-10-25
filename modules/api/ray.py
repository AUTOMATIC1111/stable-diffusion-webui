from ray import serve
import ray

from modules.api.raypi import Raypi

import time
import os


#ray.init(os.environ.get("RAY_HEAD_ADDRESS", ""))
#ray.init("ray://localhost:10001")



def ray_only():
    serve.shutdown()
    
    print("starting ray")
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


if __name__ == "__main__":

    if "RAY_DOCKER" in os.environ:
        ray.init(
            dashboard_host=os.environ.get("RAY_DASHBOARD_HOST", "0.0.0.0"),
            dashboard_port=int(os.environ.get("RAY_DASHBOARD_PORT", 8265))
            )
    elif "RAY_HEAD_ADDRESS" in os.environ:
        ray.init(os.environ.get("RAY_HEAD_ADDRESS", ""))
    elif "RAY_SERVE_ONLY" in os.environ:
        ray_only()
    else:
        ray.init()