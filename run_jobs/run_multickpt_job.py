"""Submit a cloud job: ``python run.py 03_spatad_cycle_multickpt`` (one YAML, see ``run_multickpt.sh``)."""

import client_lib

REGION = "SR008"
INSTANCE_TYPE = "a100plus.8gpu.80vG.96C.1456G"
N_WORKERS = 3
# BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.3-torch2-py310:0.0.37"
BASE_IMAGE = "cr.ai.cloud.ru/5abd3ca5-ad02-487c-8516-197be6c1b5ac/dl3dv:v2"

job = client_lib.Job(
    base_image=BASE_IMAGE,
    script="bash /home/jovyan/users/shirokov/videogen_data/run_jobs/run_multickpt.sh",
    region=REGION,
    instance_type=INSTANCE_TYPE,
    n_workers=N_WORKERS,
    type="binary",
    processes_per_worker=1,
    job_desc="full dl3dv",
)

print(job.submit())
