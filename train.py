from ultralytics import YOLO
import torch

# torch.backends.cudnn.enabled = False

model = YOLO('yolov8n.pt')
# model = YOLO('yolov5l.pt', data="C:\Users\Gamer\PycharmProjects\pythonProject\Dataset\SplitData\dataOffline.yaml", batch_size=8)  # Set the desired batch size (e.g., 8)
# model = YOLO('yolov5l.pt', data="C:\\Users\\Gamer\\PycharmProjects\\pythonProject\\Dataset\\SplitData\\dataOffline.yaml", batch_size=8)
# model = YOLO('yolov5l.pt', data=r"C:\Users\Gamer\PycharmProjects\pythonProject\Dataset\SplitData\dataOffline.yaml", batch_size=8)


def main():
    model.train(data='Dataset/SplitData/dataOffline.yaml', epochs=10, batch = 2)


if __name__ == '__main__':
    main()
