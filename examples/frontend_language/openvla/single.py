import numpy as np
from token2action import TokenToAction, image_qa

import sglang as sgl

converter = TokenToAction()


def single():
    state = image_qa.run(
        image_path="images/robot.jpg",
        question="In: What action should the robot take to {<INSTRUCTION>}?\nOut:",
        max_new_tokens=7,
        temperature=0,
    )
    output_ids = state.get_meta_info("action")["output_ids"]
    print(output_ids)
    assert output_ids == [31888, 31869, 31900, 31912, 31823, 31882, 31744]
    action = converter.convert(output_ids)
    assert np.array_equal(
        np.round(action, 5),
        np.round(
            [
                -3.78757518e-03,
                5.47156949e-04,
                -2.41243806e-04,
                -2.50440557e-02,
                2.53441257e-02,
                -1.77964902e-02,
                9.96078431e-01,
            ],
            5,
        ),
    )
    return action


if __name__ == "__main__":
    runtime = sgl.Runtime(
        model_path="openvla/openvla-7b",
        tokenizer_path="openvla/openvla-7b",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
    )
    sgl.set_default_backend(runtime)
    ouput_ids = single()
    print(ouput_ids)
    runtime.shutdown()
