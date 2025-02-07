import numpy as np
import sglang as sgl
from token2action import TokenToAction, image_qa
import pandas as pd
import json
import os

converter = TokenToAction()

def batch(batch_size, temp):
    INSTRUCTION = "place the watermelon on the towel"
    # Create prompt
    SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    
    def get_openvla_prompt(instruction: str) -> str:
        return f"USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
    prompt = get_openvla_prompt(INSTRUCTION)
    arguments = [
        {
            "image_path": "images/test_obs.jpg",
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
    #CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path Embodied-CoT/ecot-openvla-7b-bridge --trust-remote-code --port 30000 --disable-radix-cach
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    batch_size = 10
    temperature = 0.2
    actions = batch(batch_size=batch_size, temp=temperature)
    print(actions)