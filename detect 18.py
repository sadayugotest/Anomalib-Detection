if __name__ == '__main__':

    import os
    os.environ['HF_HUB_OFFLINE'] = '1'

    import torch
    import cv2
    import matplotlib.pyplot as plt
    from anomalib.models import Padim
    from anomalib.engine import Engine
    from anomalib.data import Folder  # ใช้ Folder เหมือนตอนเทรน

    # 1. ระบุตำแหน่งไฟล์
    checkpoint_path = r"C:/Users/c028479/PycharmProjects/pythonProject/Anomalib_new/model3.ckpt"
    # ชี้ไปที่โฟลเดอร์ที่มีภาพที่ต้องการตรวจ (Anomalib มักต้องการ Folder root)
    dataset_root = r"C:/Users/c028479/PycharmProjects/pythonProject/Anomalib_new"
    image_path = r"C:/Users/c028479/PycharmProjects/pythonProject/Anomalib_new/Ano.jpg"

    # 2. สร้างโครงสร้าง Model และ Engine
    model = Padim(backbone="resnet18", pre_trained=False)
    engine = Engine(task="classification")

    # 3. ใช้ Folder DataModule ชี้ไปที่รูปภาพ (ใช้โหมดเดียวกับตอนเทรน)
    datamodule = Folder(
        name="inference",
        root=dataset_root,
        normal_dir="normal",      # ใส่ชื่อโฟลเดอร์ให้ตรงโครงสร้างเดิม
        abnormal_dir="abnormal", 
        task="classification"
    )
    datamodule.setup() # เตรียมข้อมูล

    # 4. ทำการ Prediction โดยใช้ datamodule
    predictions = engine.predict(
        model=model, 
        ckpt_path=checkpoint_path, 
        datamodule=datamodule
    )

    # 5. จัดการผลลัพธ์ (ดึงจาก Batch แรก รูปแรก)
    # predictions คือ List[Batch] -> เข้าถึง Batch ที่ 0
    first_batch = predictions[0]

    # ดึงค่า Score และ Label (ดึงค่าแรกของ Batch มา)
    # หมายเหตุ: ถ้าในโฟลเดอร์มีหลายรูป ค่าพวกนี้จะเป็นลิสต์/เทนเซอร์ยาวๆ
    pred_score = first_batch["pred_scores"][0].item()
    pred_label = first_batch["pred_labels"][0].item()
    anomaly_map = first_batch["anomaly_maps"][0].cpu().numpy()

    # จัดการมิติ Heatmap
    while anomaly_map.ndim > 2:
        anomaly_map = anomaly_map.squeeze(0)

    print(f"\n--- Prediction Result ---")
    print(f"Anomaly Score: {pred_score:.4f}")
    print(f"Status: {'ABNORMAL' if pred_label else 'NORMAL'}")

    # 6. แสดงผล
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    # ดึงภาพต้นฉบับที่ AI ใช้ประมวลผลออกมา (ถ้ามีใน batch)
    if "image" in first_batch:
        # Anomalib เก็บภาพเป็น Tensor [C, H, W] และ Normalize มาแล้ว
        # วิธีที่ง่ายที่สุดคือใช้ cv2 อ่านตาม image_path เดิมของคุณไปก่อนได้ครับ
        img = cv2.imread(image_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.text(0.5, 0.5, "Image not in batch")

    plt.subplot(1, 2, 2)
    plt.title(f"Heatmap (Score: {pred_score:.2f})")
    plt.imshow(anomaly_map, cmap='jet')
    plt.colorbar()

    plt.tight_layout()
    plt.show()