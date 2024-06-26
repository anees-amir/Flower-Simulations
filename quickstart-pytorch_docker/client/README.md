# Flower Example using PyTorch and Docker

To build the client image, run the following docker command:

docker build -t flower-client-image -f flower-client.dockerfile .

And to run the container for the first client for the above image, run the following:

docker run -it -v "$(pwd)/../../data/data_1.csv:/app/data.csv" --name flower_client_1_cont --network host flower-client-image

For the second client:

docker run -it -v "$(pwd)/../../data/data_2.csv:/app/data.csv" --name flower_client_2_cont --network host flower-client-image

For the third client:

docker run -it -v "$(pwd)/../../data/data_3.csv:/app/data.csv" --name flower_client_3_cont --network host flower-client-image