import requests

INSTRUCTION = "place the watermelon on the towel"
prompt = "A chat between a curious user and an artificial intelligence assistant. " + \
    "The assistant gives helpful, detailed, and polite answers to the user's questions. " + \
    f"USER: What action should the robot take to {INSTRUCTION.lower()}? ASSISTANT: TASK:"
image_data = "/root/sglang-vla/cot_bench/images/test_obs.jpg" 

def repeat_string(s, batch_size):
    return [s] * batch_size

batch_size = 1
prompt = repeat_string(prompt, batch_size)
image_data = repeat_string(image_data, batch_size)

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": prompt,
        # "image_data": image_data,
        "image_data": "https://github.com/jackyk02/sglang-vla/blob/74c349a2fc47187bcf14480574e6ed29b32fc678/cot_bench/images/resized_image.jpg?raw=true",
        # "return_logprob": "True",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 2048,
        },
    },
)
print(response.json()[0]['text'])