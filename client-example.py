import asyncio
import base64
from io import BytesIO
import time

import requests
import websockets
from PIL import Image


async def listen():
    uri = "ws://localhost:7890"
    async with websockets.connect(uri) as ws:  # type: ignore
        await ws.send("dry_run")
        while True:
            message = await ws.recv()
            print(f"{message}")


async def send_image():
    uri = "ws://localhost:7890"
    async with websockets.connect(uri) as ws:  # type: ignore
        url = "http://images.cocodataset.org/test2017/000000000674.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue())
        print('message sent at ', time.time())
        await ws.send(encoded)
        message = await ws.recv()
        print('message received at ', time.time())
        depth = Image.open(BytesIO(base64.b64decode(message)))
        depth.save('depth_from_client.png')


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(send_image())
