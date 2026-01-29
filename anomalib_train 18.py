import os
# สั่งให้ห้ามออกเน็ตเด็ดขาด ให้ดูแค่ในเครื่องเท่านั้น
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torchvision.models as models
from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine

backbone_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


# --- ส่วนที่เหลือของ Code เหมือนเดิม ---
datamodule = Folder(
    name="my_inspection",
    root="dataset_L", 
    normal_dir="normal",
    abnormal_dir="abnormal",
    task="classification"
)



model = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
    pre_trained=False # ปิดการโหลดอัตโนมัติ
)


engine = Engine(task="classification")

if __name__ == "__main__":
    engine.fit(model=model, datamodule=datamodule)