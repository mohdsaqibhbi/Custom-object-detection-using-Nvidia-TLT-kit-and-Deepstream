# Custom-object-detection-using-Nvidia-TLT-kit-and-Deepstream

This repo is about how to train a custom object detection model (using DetectNet_v2 architecture) on our own dataset using Nvidia Transfer Learning Toolkit (TLT-kit) and AWS GPU instance. Later, using the trained model to do inferencing on Jetson nano device with the help of Nvidia Deepstream SDK.

## Brief Introduction

- ### Transfer Learning Toolkit (TLT-kit)
    The NVIDIA TLT-kit is used with NVIDIA pre-trained models to create custom Computer Vision (CV) and Conversational AI models with the user’s own data. Training AI models using TLT-kit does not require expertise in AI or deep learning.
- ### Deepstream SDK
    NVIDIA's DeepStream SDK delivers a complete streaming analytics toolkit for AI-based multi-sensor processing, video, audio and image understanding. DeepStream is for vision AI developers, software partners, startups and OEMs building IVA apps and services.


## Getting Started

### Prerequisites
- Hands-on practice in AWS instance.
- AWS GPU instance (g4dn.xlarge) as a training machine.
- An NGC account to download Nvidia pre-trained models.

### Steps for training the model

1. Connect to an AWS GPU instance.
2. Make nested directories

    ```mkdir -p tlt_experiments/data tlt_experiments/detectnet_v2```

3. Move the dataset to the ```tlt_experiments/data/``` folder. Make sure the dataset is in KITTI format. Follow [this](https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html#id3) to make your dataset in KITTI format.
4. Set up an NGC account, generate API KEY and download Command Line Interface (CLI)
    - Create an NGC account [here](https://ngc.nvidia.com/signin) and sign in.
    - Generate an API KEY [here](https://ngc.nvidia.com/setup) and save the KEY somewhere.
    - Download CLI by following the instructions [here](https://ngc.nvidia.com/setup).
    
    *For more information follow [this](https://www.youtube.com/watch?v=pCGc_sybX-s&t=379s&ab_channel=joevvaldivia).*

5. Set up a TLT container for training. Optionally, upgrade the python version if required from [here](https://dev.to/serhatteker/how-to-upgrade-to-python-3-7-on-ubuntu-18-04-18-10-5hab). Also make sure the port 8888 is enabled in the EC2 instance.
    - Pull TLT-kit docker image using 
        
        ```docker pull nvcr.io/nvidia/tlt-streamanalytics:v2.0_py3```
    - Run docker container using 
        
        ```docker run --runtime=nvidia -it -v /home/ubuntu/tlt-experiments:/workspace/tlt-experiments -p 8888:8888 nvcr.io/nvidia/tlt-streamanalytics:v2.0_py3 /bin/bash```
    
    - You will end up in the container.
6. To start the container later follow this.
    - To start the TLT container
        
        ```docker start <container id>``` 
    - To attach the TLT container
        
        ```docker attach <container id>```

7. Install TLT-kit and check if it’s successfully installed.
    - Install TLT-kit by running the following.
        
        ```pip3 install nvidia-pyindex```

        ```pip3 install nvidia-tlt```
    
    - Check if it’s successfully installed.
        
        ```tlt --help```

8. (Optionally) Install and configure the jupyter notebook if it's not already by following [this](https://www.youtube.com/watch?v=qYe5J5lBvn4&t=396s&ab_channel=SrceCde).

9. Open jupyter notebook
    - To open the jupyter notebook run the following command.
        
        ```jupyter notebook --ip 0.0.0.0 --no-browser --port 8888 --allow-root```
    
    - Copy-paste the URL to the browser and replace the localhost IP in the URL with the public IP of the EC2 instance.


10. Goto ```examples/detectnet_v2``` directory and run the notebook. Step-by-step follow **[detectnet_v2.ipynb](examples/detectnet_v2.ipynb)** notebook for training. Follow this for TLT kit documentation.
11. While exporting the model for Deepstream on a Jetson nano device make sure it’s FP32 or FP16. Jetson nano device doesn’t support INT8.

### Steps for running the model on Jetson

1. Once you get the **.etlt** model, generate a **labels.txt** file containing the class names. Follow instructions from [here](https://docs.nvidia.com/tao/archive/tlt-20/tlt-user-guide/text/deploying_to_deepstream.html#integrating-a-detectnet-v2-model) for more details.
2. Move **.etlt** model and **labels.txt** file in **deepstream-python-apps** directory on Jetson nano device.
3. Use **deepstream-test3** app and configure the **dstest3_pgie_config.txt** by following the instructions from [here](https://docs.nvidia.com/tao/archive/tlt-20/tlt-user-guide/text/deploying_to_deepstream.html#integrating-a-detectnet-v2-model).
Set API KEY in the **dstest3_pgie_config.txt** as
    ```tlt-model-key=tlt_encode```

4. Run deepstream-test3 by following [this](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-test3) and see the output.

## Demo

I have trained an object detection model to detect the defect on the surface of metal objects. This type of models can be used in any manufacturing industries for quality check purpose of metal parts. Here is the inference output demo of our Defect Detection model.

[![Sample Output]()](https://youtu.be/7NOCFuCDX0A)

## References

- [Nvidia TLT Quick Start Guide](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/tlt_quick_start_guide.html)
- [Nvidia Deepstream SDK](https://developer.nvidia.com/deepstream-sdk#:~:text=NVIDIA's%20DeepStream%20SDK%20delivers%20a,building%20IVA%20apps%20and%20services.)
