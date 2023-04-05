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
python simulation.py
```

The simulation will use the `sim_config.json` file in the project's parent directory for its configuration settings. The following is an example of a proper `sim_config.json`:

```json
{
    "server_strategy": "standard",
    "n_clients": 10,
    "n_bad_clients": 1,
    "n_scrambled_labels": 4,
    "n_rounds": 1,
    "dataset": "mnist",
    "n_client_epochs": 5,
    "client_batch_size": 128,
    "avg_client_std_threshold": 1.0
}
```

For a full research paper covering this project, please see `paper.md`.
