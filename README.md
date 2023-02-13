# Sewer Defect Detection System
 This repository contains the back-end code for the Sewer Defect Detection System as implemented at Royal HashkoningDHV. The system is developed to increase the efficiency and reliability of second-line cctv inspections of sewer infrastructure. It employs a deep learning object detection model (YOLOv5) to detect three types of defects (cracks, roots and displaced joints) and several filter methods to select the most relevant frame for each potential defect. 

## Installation 
 To install the system, clone this repository and install [requirements.txt](https://github.com/jeannot-github/Sewer-Defect-Detection-System/blob/main/requirements.txt).

 ```bash
 git clone https://github.com/jeannot-github/Sewer-Defect-Detection-System # clone repository
 cd Sewer-Defect-Detection-System # change directory
 pip install -r requirements.txt # install requirements
 ```

 Preferably, you should also use CUDA and cuDNN for GPU acceleration. 

 ## Usage
  The execute the system on new cctv footage, set the variables in [config.yaml](https://github.com/jeannot-github/Sewer-Defect-Detection-System/blob/main/config.yaml) and execute [main.py](https://github.com/jeannot-github/Sewer-Defect-Detection-System/blob/main/main.py).

  ```bash
 python main.py # execute 
 ```

 Use the boolean variable LOCAL in [config.yaml](https://github.com/jeannot-github/Sewer-Defect-Detection-System/blob/main/config.yaml) to load yolov5 locally or via the yolov5 Github (default is False). 