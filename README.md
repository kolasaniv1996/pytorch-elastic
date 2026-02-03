# pytorch-elastic

# Elastic Training with PyTorch and Run:AI

## Overview

Elastic training in PyTorch allows for dynamic handling of worker nodes in a distributed training job. This ensures minimal interruption and maximizes resource utilization even in cases of worker failures or membership changes.

## Elastic Training Flow

1. **Initiating the Elastic Run**:
    - Specify the minimum and maximum number of workers using `--nnodes=MIN_SIZE:MAX_SIZE`.
    - Define the number of allowed failures or membership changes with `--max-restarts=NUM_ALLOWED_FAILURES_OR_MEMBERSHIP_CHANGES`.

2. **Handling Membership Changes or Worker Failures**:
    - If a worker becomes unavailable (e.g., due to network issues or reclaimed resources):
      - PyTorch will kill and restart the worker group to rebalance the workload.
      - It will check the number of available workers, rerank them, load the latest checkpoints, and resume training with the remaining nodes.

3. **Restart Conditions**:
    - The process will continue smoothly as long as the `--max-restarts` value is not exceeded.
    - Dynamic handling ensures minimal interruption and maximizes resource utilization.

### Important Note
Checkpointing is a crucial part of Torch Elastic :
> "On failures or membership changes, ALL surviving workers are killed immediately. Make sure to checkpoint your progress. The frequency of checkpoints should depend on your job’s tolerance for lost work."

## Do I Need to Change My Training Script?

No, you can use the same training scripts. However, if you are not usually checkpointing your progress, you need to start saving those. Make sure to include `load_checkpoint(path)` and `save_checkpoint(path)` logic in your script.


Launching the Distributed Training on Run:ai
To start the elastic distributed training on Run:ai, ensure that you have the correct version of the CLI (v2.17 or later). To launch a distributed PyTorch training job, use the runai submit-dist pytorch command depending on your CLI version. Here is the command to submit our job:

runai training pytorch submit adobe-workflow \
  --image vivekkolasani1996/adobe-elastic-pytorch-satya:v2 \
  --workers 2 \
  --port service-type=NodePort,container=8080 \
  --run-as-user \   
  --command -- bash -c "/app/launch.sh && sleep infinity"


