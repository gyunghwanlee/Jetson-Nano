# Jetson-Nano

1. Anaconda 설치
2. Python 3.8 가상환경 생성
3. 가상환경에서 pytorch, torchvision (Nano에선 jetpack 4.6까지 지원, 기존대로라면 CUDA 호환이 불가하나 링크를 통해 Jetpack 5 기반 pytorch, torchvision 설치)
   pip install -U pip wheel gdown
   gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
   gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV
   pip install torch파일명 torchvision파일명

   torch, torchvision import하여 버전 확인 후 CUDA 사용 여부도 확인, erorr 발생 시에 numpy 버전 업그레이드

4. Realsense SDK 설치 (https://github.com/jetsonhacks/installRealSenseSDK)
   bashrc 수정, 파일 경로 수정 유의

5. 가상환경에서 git clone 및 필요 파일들 설치
   sudo git clone https://github.com/gyunghwanlee/Jetson-Nano
   cd Jetson-Nano
   pip install ultralytics
   pip install -r requirements.txt
   


