# TensorFlow GPU 설정

0. Tensorflow GPU 인식 확인

1. NVIDIA GPU driver 설치
2. CUDA Toolkit 설치
3. cuDNN 다운로드
4. zlibwapi.dll (optional)

## 0. Tensorflow GPU 인식 확인

python3 인터프리터로 tensorflow GPU 인식 확인

```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

```
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 14721738066926660773
xla_global_id: -1
]
```

GPU를 인식하지 못하므로 다음 단계 진행

## 1. NVIDIA GPU driver 설치

### 그래픽카드 정보 및 드라이버 확인

아래 세 가지 명령어중 한 가지만 터미널에 입력 후 엔터

(세 번째 것으로 진행)

```
ubuntu-drivers devices
```

```
lshw -numeric -C display
```

```
lspci | grep -i nvidia
```

```
oem@user:~$ lspci | grep -i nvidia
3a:00.0 3D controller: NVIDIA Corporation GP108M [GeForce MX150] (rev a1)
```

### 드라이버 설치

* 권장 드라이버 자동으로 설치

  ```
  sudo ubuntu-drivers autoinstall
  ```

* 원하는 버전 수동으로 설치
  ```
  sudo apt install nvidia-driver-[version num]
  ```

* 리부팅

  ```
  sudo reboot
  ```

### PPA저장소를 사용하여 자동 설치

cf) : PPA란?

[참고자료](https://velog.io/@fj2008/%EB%A6%AC%EB%88%85%EC%8A%A4-ubuntu-PPA-%ED%8D%BC%EC%8A%A4%EB%84%90-%ED%8C%A8%ED%82%A4%EC%A7%80-%EC%95%84%EC%B9%B4%EC%9D%B4%EB%B8%8C%EA%B0%9C%EC%9D%B8%EC%A0%80%EC%9E%A5%EC%86%8C)

> 리눅스는 업데이트를 각 프로그램이 직접 하는 것이 아닌 패키지 저장소를 이용하여 업데이트를 해야한다.
>
> 하지만 우분투 공식 패키지 저장소에서는 유명한 프로그램이 아닌 일반 프로그램의 최신 버전이 담겨있지 않기에 이러한 업데이트/설치를 PPA에서 할 수 있게 된 것이다.
>
> 즉, 개인 패키지 저장소란 뜻을 가진 PPA는 런치패드에서 제공하는 우분투의 공식 패키지 저장소에 없는 서드 파티 소프트웨어를 위한 개인용 소프트웨어 패키지 저장소이다.
>
> 쉽게 풀어서 설명하자면 우분투의 공식 패키지 저장소에 없는 소프트웨어 및 버전을 직접 찾아서 개인저장소에 추가하는 방법이다.
>
> 

PPA 저장소를 사용하면 최신 버전의 베타 그래픽 드라이버를 설치할 수 있다. 아래 명령으로 Graphics-Drivers/PPA 저장소를 시스템에 추가한다.

```
sudo add-apt-repository ppa:graphics-drivers/ppa
```

```
sudo apt update
```

## 2. CUDA Toolkit 설치

`nvidia-smi` 명령어로 확인한 CUDA version에 맞게 CUDA toolkit을 설치한다. 

아래 링크에서 설치할 수 있다.

https://developer.nvidia.com/cuda-toolkit-archive

```
oem@user:~$ nvidia-smi
Tue Apr 16 14:43:10 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce MX150           Off | 00000000:3A:00.0 Off |                  N/A |
| N/A   43C    P8              N/A /  14W |      9MiB /  2048MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1039      G   /usr/lib/xorg/Xorg                            4MiB |
|    0   N/A  N/A      8306      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
```

알맞은 버전이 12.2 버전이므로 12.2 버전을 설치한다.

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
```

```
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
```

```
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.2-535.104.05-1_amd64.deb
```

```
sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.2-535.104.05-1_amd64.deb
```

```
sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
```

```
sudo apt-get update
```

```
sudo apt-get -y install cuda
```

## 3. cuDNN 다운로드

cuDNN(CUDA Deep Neural Network)는 아래 링크에서 다운 받을 수 있다.

https://developer.nvidia.com/rdp/cudnn-download

```
wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2004-9.1.0_1.0-1_amd64.deb
```

```
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.1.0_1.0-1_amd64.deb
```

```
sudo cp /var/cudnn-local-repo-ubuntu2004-9.1.0/cudnn-*-keyring.gpg /usr/share/keyrings/
```

```
sudo apt-get update
```

```
sudo apt-get -y install cudnn
```

CUDA 11(또는 12)을 설치하기 위해서는 CUDA 11(또는 12) 세부 패키지를 설치해야 한다.

```
sudo apt-get -y install cudnn-cuda-11
```

```
sudo apt-get -y install cudnn-cuda-12
```

