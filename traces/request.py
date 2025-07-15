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
        "image_data": "https://raw.githubusercontent.com/MichalZawalski/embodied-CoT/main/test_obs.png",
        # "return_logprob": "True",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 2048,
        },
    },
)
print(response.json()[0]['text'])