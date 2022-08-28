# Soft_Gated_Pose_Estimation_Pytorch
This repository provides an unofficial pytorch implementation of the paper ["Toward fast and accurate human pose estimation via soft-gated skip connections"](https://arxiv.org/abs/2002.11098) *(by Adrian Bulat, Jean Kossaifi, Georgios Tzimiropoulos, Maja Pantic)*.

Some code for the data preparation and the stacked hourglass network implementation is taken from this repository: [pytorch_stacked_hourglass](https://github.com/princeton-vl/pytorch_stacked_hourglass).

A live demo flask web application can be found in this [repository](https://github.com/dkurzend/Human_Pose_Estimation_Flask_App).

### Dataset
- [x] MPII Human Pose

### Models
- [x] Stacked Hourglass Network
- [x] Soft-Gated Skip Connections



## Getting Started

### Requirements
- Python 3.10.4
- CUDA available (tested with Cuda 10.2)
- For training 4 GPU's have been used
- If no 4 GPU's are available, remove line 127 in train.py

### Installation
1. Clone the repository:

    ```
    git clone https://github.com/dkurzend/Soft_Gated_Pose_Estimation_Pytorch.git
    ```
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and create a virtual environment.
    ```
    conda create --name hpe_env
    ```

3. Activate the virtual env and install pip as well as the dependencies.
    ```
    conda activate hpe_env
    conda install pip
    pip install -r requirements.txt
    ```
    (Alternatively use venv instead of miniconda)

4. Download the full [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de/), and place the `images` directory in `data/mpii/`

5. in config.py are the configurations that I used to train the soft-gated skip connections network and the stacked hourglass network. Comment out the configuration you dont need or create your own configuration.

## Training
Before training you have to create a directory `checkpoints` in project root directory. Now you can train the model with
    ```
    python train.py
    ```
    Training might take 3-4 days.

### Pretrained Models

- A trained soft-gated skip connections model can be downloaded [here](https://matix.li/4704c467c50a).
- A trained stacked hourglass model can be downloaded [here](https://matix.li/9680ec1eb999).

Download them and put them into the `checkpoints` folder.

Both models were trained for 200 epochs using the configuration in config.py. The chosen batch size is 24.
As optimizer Adam is used instead of RMSProp which is proposed in both papers.
The learning rate varies from 2.5e-4 and 1e-5 dropping at epochs 75, 100 and 150 in a linear manner. The results of both models are presented [below](#results).


### Validation and Inference

For validation run `python validation.py`

For inference of a sinlge image run `python inference.py`. In config.py under `config['inference']['presentation_dir']` you can change the directory to store the resulting images.
Both papers use for the final prediction the average of the original image and the flipped image. However our goal was to test the soft-gated skip connections network in a real-time web application using the webcam of the computer. Therefore we omitted this step in favor of efficiency accepting that the accuracy will drop compared to the results of both papers.


## Results


![Example prediction](/presentation/img_with_keypoints.png)            |  ![Example prediction](/presentation/kid_with_kp.png)
:-------------------------:|:-------------------------:




### Stacked Hourglass Network
<p>Val PCK @, 0.5 , total : 0.875 , count: 44239<br>
Val PCK @, 0.5 , ankle : 0.772 , count: 4234<br>
Val PCK @, 0.5 , knee : 0.813 , count: 4963<br>
 Val PCK @, 0.5 , hip : 0.843 , count: 5777 <br>
Val PCK @, 0.5 , pelvis : 0.886 , count: 2878<br>
Val PCK @, 0.5 , thorax : 0.979 , count: 2932<br>
Val PCK @, 0.5 , neck : 0.975 , count: 2932<br>
Val PCK @, 0.5 , head : 0.948 , count: 2931<br>
Val PCK @, 0.5 , wrist : 0.82 , count: 5837<br>
Val PCK @, 0.5 , elbow : 0.879 , count: 5867<br>
Val PCK @, 0.5 , shoulder : 0.938 , count: 5888 </p>

### Soft-Gated Skip Connections
<p>Val PCK @, 0.5 , total : 0.874 , count: 44239<br>
Val PCK @, 0.5 , ankle : 0.77 , count: 4234<br>
Val PCK @, 0.5 , knee : 0.805 , count: 4963<br>
Val PCK @, 0.5 , hip : 0.852 , count: 5777<br>
Val PCK @, 0.5 , pelvis : 0.899 , count: 2878<br>
Val PCK @, 0.5 , thorax : 0.975 , count: 2932<br>
Val PCK @, 0.5 , neck : 0.973 , count: 2932<br>
Val PCK @, 0.5 , head : 0.946 , count: 2931<br>
Val PCK @, 0.5 , wrist : 0.813 , count: 5837<br>
Val PCK @, 0.5 , elbow : 0.877 , count: 5867<br>
Val PCK @, 0.5 , shoulder : 0.941 , count: 5888</p>


## Final Note
This repository was part of a university project at university of Tübingen.
Project team:<br>
David Kurzendörfer, Jan-Patrick Kirchner, Tim Herold, Daniel Banciu
