import pyrealsense2 as rs  # Intel RealSense cross-platform SDK
import numpy as np  # 넘파이 라이브러리, 배열 연산에 사용

class RealSenseCamera:
    def __init__(self):
        # RealSense SDK의 파이프라인 초기화
        self.pipeline = rs.pipeline()
        # 파이프라인 설정 객체 생성
        self.config = rs.config()
        # 컬러 스트림 활성화, 해상도 640x480, BGR 포맷, 30fps
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # 깊이 스트림 활성화, 해상도 640x480, 16비트, 30fps
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # 설정된 파이프라인 시작
        self.pipeline.start(self.config)

    def get_frames(self):
        """카메라로부터 프레임을 취득하여 컬러 이미지와 깊이 이미지로 반환합니다."""
        # 파이프라인에서 프레임 세트를 기다림
        frames = self.pipeline.wait_for_frames()
        # 프레임 세트로부터 컬러 프레임을 얻음
        color_frame = frames.get_color_frame()
        # 프레임 세트로부터 깊이 프레임을 얻음
        depth_frame = frames.get_depth_frame()
        # 컬러 프레임을 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        # 깊이 프레임을 numpy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image  # 컬러 및 깊이 이미지 반환

    def release(self):
        """카메라 리소스를 정리합니다."""
        # 파이프라인 정지
        self.pipeline.stop()
