import numpy as np
import sglang as sgl
from token2action import TokenToAction, image_qa
import pandas as pd
import json
import os

converter = TokenToAction()

def batch(batch_size, temp):
    instruction = "close the drawer"
    prompt = "A chat between a curious user and an artificial intelligence assistant. " + \
        "The assistant gives helpful, detailed, and polite answers to the user's questions. " + \
        f"USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
        
    arguments = [
        {
            "image_path": "images/robot.jpg",
            "question": prompt,
        }
    ] * batch_size
    states = image_qa.run_batch(
        arguments,
        max_new_tokens=1024,
        temperature=temp
    )
    return [{'cot': s.text(), 'action': s.get_meta_info("action")["output_ids"][-8:-1]} for s in states]


if __name__ == '__main__':
    #python -m sglang.launch_server --model-path openvla/openvla-7b --trust-remote-code --quantization fp8 --port 30000 --disable-radix-cach --dp 2 --tp 2
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    batch_size = 2
    temperature = 0.1
    actions = batch(batch_size=batch_size, temp=temperature)
    print(actions)