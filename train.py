from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
from ultralytics.utils import LOGGER
import os
import torch
import yaml

os.environ["PYTHONHASHSEED"] = "0"

data = "./dataset/data.yaml"
model_path = "./yoloe-26m-seg.pt"
pe_path = "./coco-pe.pt"
s
with open(data, "r") as f:
    dataset_cfg = yaml.safe_load(f)

names = dataset_cfg["names"]

if isinstance(names, dict):
    names = list(names.values())

LOGGER.info(f"Classes: {names}")

model = YOLOE(model_path)

tpe = model.get_text_pe(names)
torch.save(
    {
        "names": names,
        "pe": tpe,
    },
    pe_path,
)

head_index = len(model.model.model) - 1

freeze = [str(i) for i in range(head_index)]

for name, child in model.model.model[-1].named_children():
    if "cv3" not in name:
        freeze.append(f"{head_index}.{name}")

freeze.extend(
    [
        f"{head_index}.cv3.0.0",
        f"{head_index}.cv3.0.1",
        f"{head_index}.cv3.1.0",
        f"{head_index}.cv3.1.1",
        f"{head_index}.cv3.2.0",
        f"{head_index}.cv3.2.1",
    ]
)

LOGGER.info(f"Frozen layers: {freeze}")
LOGGER.info(f"Prompt embeddings saved to: {pe_path}")

model.train(
    data=data,
    epochs=100,
    close_mosaic=5,
    batch=1,
    optimizer="AdamW",
    lr0=1e-3,
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=4,
    device="0",
    trainer=YOLOEPESegTrainer,
    freeze=freeze,
)

