import cv2, onnxruntime, copy
from ultralytics import YOLO
from PIL import Image
from yolox import YoloxONNX

class YOLOv8:
    def __init__(self, model_path, classes=[0, 2], conf=0.5):
        self.yolo = YOLO(model_path)
        self.classes = classes
        self.conf = conf
        print(self.yolo.names)

    def pred(self, img):
        self.results = self.yolo(img, conf=self.conf, classes=self.classes)
        self.boxes = self.results[0].boxes.xyxy.cpu().numpy()
        self.labels = self.results[0].boxes.cls.cpu().numpy()
        self.logits = self.results[0].boxes.conf.cpu().numpy()
        return self.boxes, self.logits, self.labels
    
    def get_img(self):
        img = self.results[0].orig_img
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = [int(x) for x in box]
            label = self.results[0].names[self.labels[i]] + ' ' + str(self.logits[i])
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            img = cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img  
    
def main():
    # Load image
    video_path = 'SampleVideo_LowQuality.mp4'
    cap = cv2.VideoCapture(video_path)

    # ここでfpsを設定すると、動画の再生速度が変わる
    cap.set(cv2.CAP_PROP_FPS, 30)
    assert cap.isOpened(), f'Failed to load video {video_path}'

    # Load model
    model_path = 'models/yolov8n.pt'
    conf = 0.5
    classes = [0, 2]
    yolo = YOLOv8(model_path, classes=classes, conf=conf)

    model_path = 'models/yolox_nano.onnx'
    yolo = YoloxONNX(model_path=model_path)

    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        if not ret:
            break

        # Inference
        results = yolo.pred(frame)

        # cv2.imshow('frame', results[0].plot())
        cv2.imshow('frame', yolo.get_img())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == '__main__':
    main()