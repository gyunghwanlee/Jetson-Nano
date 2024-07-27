import cv2
import time
import os
import csv
from realsense_depth import RealSenseCamera # Intel Realsense 카메라를 사용하기 위한 클래스
from yolo_detection import YOLODetector    # YOLO 객체 탐지 모델을 사용하기 위한 클래스

def get_next_filename(base_name, ext):
    """기존 파일 이름과 중복되지 않도록 새로운 파일 이름을 생성합니다."""
    counter = 1  # 파일 이름에 사용할 카운터
    filename = f"{base_name}{counter}.{ext}"  # 기본 파일 이름 생성
    while os.path.exists(filename):  # 해당 이름의 파일이 이미 존재할 경우
        counter += 1  # 카운터 증가
        filename = f"{base_name}{counter}.{ext}"  # 새 파일 이름 생성
    return filename  # 최종적으로 결정된 파일 이름 반환

def main():
    # RealSense 카메라 및 YOLO 모델 초기화
    camera = RealSenseCamera()  # 카메라 객체 생성
    detector = YOLODetector()   # 탐지 모델 객체 생성

    # 데이터 저장을 위한 파일 이름 설정
    base_filename = "data"
    filename = get_next_filename(base_filename, 'csv')  # 중복 없는 파일명 생성
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)  # CSV 파일 작성을 위한 writer 객체 생성

        # CSV 파일 헤더 작성
        writer.writerow([
            'Timestamp', 'Class Name', 'Confidence', 
            'Bounding Box (x1, y1, x2, y2)', 'Center (x, y)', 'Depth (mm)'
        ])

        try:
            while True:
                # 프레임 캡처 및 객체 탐지
                color_image, depth_image = camera.get_frames()  # 카메라로부터 컬러 및 깊이 이미지 획득
                results = detector.detect_objects(color_image, conf_thres=0.5)  # 컬러 이미지에서 객체 탐지 수행
                detections = detector.get_class_names(results)  # 탐지 결과로부터 클래스 이름 추출

                # 현재 시간 기록
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")

                # 데이터 기록
                for detection in detections:
                    x1, y1, x2, y2 = map(int, detection['box'])  # 바운딩 박스 좌표 추출 및 정수 변환
                    class_name = detection['class_name']  # 클래스 이름
                    confidence = detection['confidence']  # 탐지 확신도
                    
                    # 중심 좌표 계산
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # 중심 좌표의 깊이 값 추출
                    depth = depth_image[cy, cx] if (0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]) else 0
                    
                    # CSV 파일에 기록할 데이터
                    writer.writerow([
                        current_time, class_name, f"{confidence:.2f}", 
                        f"({x1}, {y1}), ({x2}, {y2})", f"({cx}, {cy})", depth
                    ])

                    # 물체 바운딩 박스와 정보 화면에 표시
                    color_image = cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} ({confidence:.2f}) Depth: {depth}mm"
                    color_image = cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 결과 이미지 표시
                cv2.imshow('RealSense', color_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 눌러 종료
                    break
                
                # CSV 파일이 일정 크기를 초과하면 새 파일 생성
                file.flush()  # 파일에 버퍼링된 데이터를 즉시 기록
                if os.path.getsize(filename) > 1024 * 1024:  # 파일 크기가 1MB를 초과할 경우
                    file.close()  # 기존 파일 닫기
                    filename = get_next_filename(base_filename, 'csv')  # 새 파일 이름 생성
                    file = open(filename, 'w', newline='')  # 새 파일 열기
                    writer = csv.writer(file)  # 새 writer 객체 생성
                    writer.writerow([
                        'Timestamp', 'Class Name', 'Confidence', 
                        'Bounding Box (x1, y1, x2, y2)', 'Center (x, y)', 'Depth (mm)'
                    ])

        finally:
            camera.release()  # 카메라 리소스 해제
            cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
            file.close()  # 파일 닫기

if __name__ == "__main__":
    main()
