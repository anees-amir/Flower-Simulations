# Flower Example using PyTorch and Docker

To build the server image, run the following docker command:

docker build -t flower-server-image -f server_flower.dockerfile .

And to run the container for the above image, run the following:

docker run -it --name flower_server_cont --network host flower-server-image