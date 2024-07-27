from ultralytics import YOLO  # YOLO 객체 탐지 모델 관리를 위한 ultralytics 라이브러리
import torch  # 파이토치 라이브러리, 텐서 연산 및 딥러닝에 사용

class YOLODetector:
    def __init__(self):
        # 모델 초기화 시 사용할 YOLO 모델 파일 지정
        self.model = YOLO('yolov8n.pt')  # yolov8n 모델 파일을 로드, 다른 YOLO 모델로 변경 가능
        self.class_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
        19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
        24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
        29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
        33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
        37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
        41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
        65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
        73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
        78: 'hair drier', 79: 'toothbrush'
        }

        print("class name:", self.class_names)

    def detect_objects(self, image,conf_thres=0.5):
        """
        인풋 이미지에 대해 객체 탐지를 수행하고 결과를 반환합니다.

        Args:
            image (numpy.ndarray): 탐지를 수행할 이미지 배열

        Returns:
            results: 탐지 결과 객체, 여기서는 이를 다시 처리하기 위해 반환만 함
        """
        results = self.model(image, conf=conf_thres)  # 모델에 이미지를 전달하고 탐지 결과 받음
        return results  # 탐지 결과 반환

    def get_class_names(self, results):
        """
        탐지 결과로부터 객체 정보를 추출하여 리스트로 반환합니다.

        Args:
            results: detect_objects 메소드에서 반환된 탐지 결과 데이터

        Returns:
            detections (list): 탐지된 객체의 정보를 포함하는 딕셔너리 리스트. 
                               각 딕셔너리는 클래스 ID, 클래스 이름,
                               신뢰도, 바운딩 박스 좌표를 포함합니다.
        """
        detections = []
        for result in results:
            # 탐지 결과 객체에서 사용 가능한 클래스 이름 얻기
            class_names = result.names if hasattr(result, 'names') else {}
            # 결과에 포함된 각 객체의 바운딩 박스 처리
            for box in result.boxes:
                class_id = int(box.cls.item())  # 바운딩 박스의 클래스 ID 출력
                class_name = self.class_names.get(class_id, f'class{class_id}')
                confidence = box.conf.item() if isinstance(box.conf, torch.Tensor) else box.conf  # 신뢰도 추출
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'box': box.xyxy[0].tolist()  # 바운딩 박스 좌표 ([x1, y1, x2, y2] 형태)
                })
        return detections  # 추출된 객체 정보 리스트 반환
