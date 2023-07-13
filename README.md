# websocket-example

This is a simple example of a websocket server and client using the [websocket](https://websockets.readthedocs.io/en/stable/index.html) library. Here we use DL model to predict the depth of an image.
HuggingFace [transformers](https://huggingface.co/transformers/) library is used to load the model.

## Model used for example

The model used here https://huggingface.co/Intel/dpt-large

## Usage with Conda

```bash
conda create -n websockets python=3.10
conda activate websockets
pip install -r requirements.txt
python server.py
```

## Usage with Docker

```bash
sudo docker compose up --build -d
```

## Runing client example

```bash
python client-example.py
```
