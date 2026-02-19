import os
# สั่งให้ห้ามออกเน็ตเด็ดขาด ให้ดูแค่ในเครื่องเท่านั้น
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torchvision.models as models
from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine
from pathlib import Path

backbone_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# กำหนด path สำหรับบันทึกผลลัพธ์
results_dir = Path(r"C:/Users/c028479/PycharmProjects/pythonProject/Anomalib_new/results")
results_dir.mkdir(exist_ok=True)

# --- ส่วนที่เหลือของ Code เหมือนเดิม ---
datamodule = Folder(
    name="my_inspection",
    root=r"C:/Users/c028479/PycharmProjects/pythonProject/Anomalib_new/dataset", 
    normal_dir="normal",
    abnormal_dir="abnormal",
    task="classification"
)

model = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
    pre_trained=False # ปิดการโหลดอัตโนมัติ
)

# สร้าง Engine พร้อมกำหนด default_root_dir สำหรับบันทึกผลลัพธ์
engine = Engine(
    task="classification",
    default_root_dir=str(results_dir),
    max_epochs=1,
    logger=True,
    log_every_n_steps=1
)

if __name__ == "__main__":
    # Train model
    print("Starting training...")
    engine.fit(model=model, datamodule=datamodule)
    print(f"Training completed! Results saved to: {results_dir}")
    
    # Test model และสร้างผลลัพธ์รูปภาพ
    print("\nStarting testing and generating result images...")
    engine.test(model=model, datamodule=datamodule)
    print("Testing completed! Check the results folder for images.")
    
    # บันทึก model checkpoint
    checkpoint_path = results_dir / "model.ckpt"
    engine.trainer.save_checkpoint(checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")