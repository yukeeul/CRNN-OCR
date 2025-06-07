import glob

import hydra
import onnxruntime
import torch
import torchvision.transforms as T
from PIL import Image

from src.utils import Tokenizer, ctc_greedy_decode


@hydra.main(version_base=None, config_path="configs", config_name="main.yaml")
def run(cfg):
    img_size = cfg["model"]["img_size"]
    imgs_path = cfg["imgs_path"]
    model_id = cfg["model_id"]

    tokenizer = Tokenizer(**cfg["tokenizer"])

    # только локальные модели
    ort_session = onnxruntime.InferenceSession(f"models/{model_id}.onnx", providers=["CPUExecutionProvider"])
    files = glob.glob(f"{imgs_path}/*.jpg")
    transform = T.Compose(
        [
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ]
    )
    for filename in files:
        img = Image.open(filename)
        img = transform(img)[None, ...].numpy()

        onnxruntime_input = {
            input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), [img])
        }

        logits = ort_session.run(None, onnxruntime_input)[0]
        logits = torch.tensor(logits).argmax(dim=-1)

        ans = ctc_greedy_decode(logits)[0]
        ans = tokenizer.decode(ans)

        print(f"img: {filename}")
        print(f"ans: {ans}")
        print("")


if __name__ == "__main__":
    run()
