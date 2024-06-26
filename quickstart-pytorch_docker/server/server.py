from typing import List, Tuple
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

ip_add = input('Please enter the IP address of the server: ')
port = input('Please enter the port number of the server: ')

# Define strategy
strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
                  min_available_clients=3)


# Define config
config = ServerConfig(num_rounds=50)

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    server_address = f"{ip_add}:{port}"
    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )
