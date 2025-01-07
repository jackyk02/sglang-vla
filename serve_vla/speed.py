import numpy as np
import sglang as sgl
from token2action import TokenActionConverter, image_qa
import pandas as pd
import json
import os

converter = TokenActionConverter()

def batch(batch_size, temp):
    arguments = [
        {
            "image_path": "images/robot.jpg",
            "question": "In: What action should the robot take to {Grab the block}?\nOut:",
        }
    ] * batch_size
    states = image_qa.run_batch(
        arguments,
        max_new_tokens=7,
        temperature=temp
    )
    return [converter.token_to_action(s.get_meta_info("action")["output_ids"]).tolist() for s in states]


if __name__ == '__main__':
    #python -m sglang.launch_server --model-path openvla/openvla-7b --trust-remote-code --port 30000 --disable-radix-cache --dtype bfloat16
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    batch_size = 4
    temperature = 1.0
    actions = batch(batch_size=batch_size, temp=temperature)
    print(actions)
