# Client Model Verification for Federated Learning

A proof-of-concept for verifiying client models before incorporating them into a federated learning server's global model.

Uses [Flower, the friendly federated learning framework](https://flower.dev).

## Installation

Run the following commands to create a virtual environment and install into it the required packages.

```
python -m pip install virtualenv
python -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

To run the program, first ensure that the virtual environment has been activated (via the `source .venv/bin/activate` command), then run the following command:

```
python run_sim.py
```

For a full research paper covering this project, please see `paper.md`.
