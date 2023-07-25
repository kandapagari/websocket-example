# -*- coding: utf-8 -*-
import asyncio
import base64
import time
from io import BytesIO

import websockets
from PIL import Image

from models.depth_estimator import DepthEstimator

PORT = 7890  # Port to listen on

print("Starting server on port", PORT)
conntected = set()


async def echo(websocket, path):
    print("A New connection extablished")
    conntected.add(websocket)
    try:
        async for message in websocket:
            print("Message received: ", message)
            for ws in conntected:
                if ws != websocket:
                    await ws.send(f"Another client said: {message}")
    except websockets.exceptions.ConnectionClosed:  # type: ignore
        print("Connection closed")
    finally:
        conntected.remove(websocket)


async def dry_run(websocket, path):
    print("A New connection extablished")
    conntected.add(websocket)
    try:
        async for message in websocket:
            if message == "dry_run":
                ts = time.time()
                model.dry_run()
                await websocket.send(f"{message} finished in {time.time() - ts} seconds")
    except websockets.exceptions.ConnectionClosed:  # type: ignore
        print("Connection closed")
    finally:
        conntected.remove(websocket)


async def predict(websocket, path):
    print("A New connection extablished")
    conntected.add(websocket)
    try:
        async for message in websocket:
            print('message received at ', time.time())
            image = Image.open(BytesIO(base64.b64decode(message)))
            depth = model(image)
            buffered = BytesIO()
            depth.save(buffered, format="PNG")
            encoded = base64.b64encode(buffered.getvalue())
            depth = Image.open(BytesIO(base64.b64decode(encoded)))
            depth.save('depth_from_server.png')
            print('message sent at ', time.time())
            await websocket.send(encoded)
    except websockets.exceptions.ConnectionClosed:  # type: ignore
        print("Connection closed")
    finally:
        conntected.remove(websocket)


model = DepthEstimator()


if __name__ == "__main__":
    start_server = websockets.serve(predict, "localhost", PORT)  # type: ignore
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
