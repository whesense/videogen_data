import client_lib
import sys

REGION = "SR006"
INSTANCE_TYPE = "a100plus.8gpu.80vG.96C.1952G"
N_WORKERS = 1
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.3-torch2-py310:0.0.37"

job = client_lib.Job(
  base_image=BASE_IMAGE,
  script="sh /home/jovyan/users/shirokov/airi/videogen_data/run_jobs/run.sh",
  region=REGION,
  instance_type=INSTANCE_TYPE,
  n_workers=N_WORKERS,
  type="binary",
  processes_per_worker=1,
  job_desc="Cycle reconstruction, ray cluster",
)

print(job.submit())
