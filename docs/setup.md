# Development Setup

## Project

Real-Time Industrial Defect Detection System using the NEU-DET dataset.

The planned system includes dataset validation, YOLOv8 training and
evaluation, optimized video inference, a FastAPI service, PLC integration,
Prometheus/Grafana monitoring, and Docker deployment.

These are planned capabilities, not claims of completed implementation.

## Development environment

- Windows
- VS Code
- PowerShell
- Git
- Python 3.11
- Project-local virtual environment: `.venv`

## Repository

https://github.com/PShreesh7/Project-2

Each contributor works on their assigned development branch.

## Create the environment

Run from the repository root, only when `.venv` does not already exist:

```powershell
py -3.11 -m venv .venv
```

## Verify the environment

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
```

The checker validates Python, virtual-environment use, Git, repository
identity, development branch, and exclusion of `.venv` from Git.

It does not validate model accuracy, ML dependencies, dataset integrity,
camera operation, or deployment readiness.

## Dataset source

Use the annotated NEU-DET dataset from the official NEU database page:

https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/

Preserve the original dataset and its source information. Dataset validation
and preprocessing will be implemented separately.

## Artifact policy

Do not commit virtual environments, credentials, datasets, model weights,
or generated training outputs.

Record dependency versions, experiment settings, artifact locations,
and validation evidence as implementation progresses.