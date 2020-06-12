<h1 align="center"><img src=https://img.shields.io/badge/python-v3.6+-blue.svg> <img src=https://img.shields.io/badge/pytorch-v%201.5-brightgreen> <a href="https://github.com/facebookresearch/detectron2"><img src=https://img.shields.io/badge/Detectron-2-lightgrey></a> <img src=https://img.shields.io/github/repo-size/uditss03/Mask_Detector> <img src=https://img.shields.io/github/license/uditss03/Mask_Detector> <a href="https://www.linkedin.com/in/udit-sharma-662304158/"><img src=https://img.shields.io/badge/Linked-in-blue></a></h1>

<h1 align="center"> Mask Detector</h1>

<p align="center">Detecting Mask in Covid 19 situation using pytorch, detectron2 and pretrained model(mask_rcnn).</p>

## Demo

<img src="https://github.com/uditss03/Mask_Detector/blob/master/result/result_4.jpeg?raw=true" width="400" height="400"> <img src="https://github.com/uditss03/Mask_Detector/blob/master/result/result_3.jpg?raw=true" width="450" height="400">
 
## Installation
You must have <a href="https://www.anaconda.com/">anaconda python</a> installed.<br>
Run the code in terminal for installation.
1. Clone the repository.
```
 $ git clone https://github.com/uditss03/Mask_Detector.git
 ```
2. Change Directory to repository.
```
 $ cd Mask_Detector
```
3. Create virtual env and activate it.
``` 
 $ conda create -n myenv pip
 $ conda activate myenv
```
4. Download all dependencies.
```
 $ pip install -r requirements.txt
```
5. Make the Directory named 'output'.
```
 $ mkdir output
```
6. Download weights and save in output dir.

## Run the code

Detect masked vs unmasked faces
```
 $ python detect.py [image path] [result path]
```
 example``` python detect.py /test/1.jpeg /result```<br>
For running on webcam
```
 $ python cam_detect.py
 ```
 ### License
 MIT © [Udit Sharma](https://github.com/uditss03/Mask_Detector/blob/master/LICENSE)
