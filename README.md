# Food_Detection

# Aim and Objectives

# AIM

  To create a Food Detection System which will detect Food and then check healthy or junk food.

# OBJECTIVES

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera    module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• The main objective of this project is to detect food in real-time

# ABSTRACT

• A Quality of food can be detected by the live feed derived from the system’s camera.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one  another. Machine Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

• A healthy food may help to prevent certain long-term (chronic) diseases such as heart disease, stroke and diabetes.Healthy food also help to reduce risk of  developing Various Diseases.

• Eating  Healthy foods that are good for you and staying physically active may help you reach and maintain a healthy weight and improve how you feel.


# INTRODUCTION

• This project is based on a Food detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.

• This project can also be used to gather information about Quality of food, i.e., Healthy, Junk food and also Identify food.

• Food can be classified into Healthy and junk, based on the image annotation we give in roboflow it also shows the Name of food.

• In our model of food detection sometimes it becomes difficult to detect food because of Mixing of Healthy and junk foods Togather or various or due to Varieties of Junk and healthy foods.However, training our model with the images of these Mixed foods makes the model more accurate.

• Neural networks and machine learning have been used for these tasks and have obtained good results.

•  Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Floor detection as well.

# LITERATURE REVIEW :

• Eating healthy foods is the cornerstone of good health status and also lower your risk for developing health problems, on the other hand, fast food can have the opposite impact. Several pieces of evidence have revealed that fast food consumption increases the risk of metabolic syndrome by increasing triglyceride levels, triggers blood sugar and blood pressure spikes.

• According to the World Health Organisation (WHO), about 2.7 million people all over the globe succumb each year due to nutritional deficiency.

• Healthy food refers to a whole lot of fresh and natural products such as fruits, vegetables, whole grains, lean proteins and good fats that deliver your body with essential nutrients for carrying out several bodily processes, combat sickness and keep diseases at bay.

• While, junk food is a highly processed food that is made up of ‘empty’ calories foods loaded with full of saturated fat, sugar and devoid of nutrients which neither helps the body to nurture, focus and perform vital functions all through the day. 

• Junk food are highly processed foods loaded with calories, sugar and fat, however, it is devoid of essential nutrients like fibre, vitamins, minerals and antioxidants. It is believed to be a key factor in the obesity epidemic and a driving force in the development of chronic diseases. Eating Healthy food reduces the chances of getting sick from various diseases.

# Jetson Nano


![nano_img01](https://user-images.githubusercontent.com/101402562/204990571-57d8df82-58c5-46b4-b4b5-f785e78e9996.jpg)



# Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

# Proposed System

1.Study basics of machine learning and image recognition.

2.Start with implementation

 ➢ Front-end development
 ➢ Back-end development
 
3.Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify  the clarity of windows.

4.Use datasets to interpret the windows and suggest whether the windows are clear or unclean.

# METHODOLOGY

The Food detection system is a program that focuses on implementing real time food detection.

It is a prototype of a new product that comprises of the main module:

Food detection and then showing on viewfinder whether healthy or junk food .

# Food Detection Module

This Module is divided into two parts:

1] Food detection

• Ability to detect the food in any input image or frame. The output is the bounding box coordinates on the detected food.

• For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.

• This Datasets identifies food in a Bitmap graphic object and returns the bounding box image with annotation of windows present in each image.

2] Quality Detection

• Classification of the food based on whether it is Healthy or junk food.

• Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

• YOLOv5 was used to train and test our model for various classes like clean, unclean. We trained it for 100 epochs and achieved an accuracy of approximately 92%.

# Installation

# Initial Setup

Remove unwanted Applications.

    sudo apt-get remove --purge libreoffice*
    sudo apt-get remove --purge thunderbird* 
     
# Create Swap file

    sudo fallocate -l 10.0G /swapfile1
    sudo chmod 600 /swapfile1
    sudo mkswap /swapfile1
    sudo vim /etc/fstab
    
    #################add line###########
    /swapfile1 swap swap defaults 0 0
 

# Cuda Configuration

      vim ~/.bashrc
      
      
      #############add line #############
      export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
      export
      LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P
      ATH}}
     export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
  
# Udpate a System

         sudo apt-get update && sudo apt-get upgrade
          
         ################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################
         
         sudo apt install curl
         
         curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
         
         sudo python3 get-pip.py
         
         sudo apt-get install libopenblas-base libopenmpi-dev
         
         sudo apt-get install python3-dev build-essential autoconf libtool pkg-config python-opengl
         python-pil python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer
         libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-
         qt4 python-qt4-gl libgle3 python-dev libssl-dev libpq-dev python-dev libxml2-dev libxslt1-
         dev libldap2-dev libsasl2-dev libffi-dev libfreetype6-dev python3-dev
         
         vim ~/.bashrc
         
         ####################### add line ####################

         export OPENBLAS_CORETYPE=ARMV8
         
         source ~/.bashrc
         
         sudo pip3 install pillow
         
         curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
         
         mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
         
         sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
         
         sudo python3 -c "import torch; print(torch.cuda.is_available())"
         
# Installation of torchvision
   
         git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
         cd torchvision/
         sudo python3 setup.py install
         
# Clone yolov5 Repositories and make it Compatible with Jetson Nano.

         cd
         git clone https://github.com/ultralytics/yolov5.git
         cd yolov5/
          
         sudo pip3 install numpy==1.19.4
         history
         ##################### comment torch,PyYAML and torchvision in requirement.txt##################################
         sudo pip3 install --ignore-installed PyYAML>=5.3.1
         sudo pip3 install -r requirements.txt
         sudo python3 detect.py
         sudo python3 detect.py --weights yolov5s.pt --source 0
         
# Food Dataset Training

# We used Google Colab And Roboflow
  train your model on colab and download the weights and past them into yolov5 folder link of project

# Running Food Detection Model
 source '0' for webcam

        !python detect.py --weights best.pt --img 416 --conf 0.1 --source 0         
  
# DEMO :





https://user-images.githubusercontent.com/101402562/205008562-738e05c3-4b01-4238-a576-b5331941ca9c.mp4


# ADVANTAGES 

• This food detection system is very useful for peoples for choosing Good and healthy food to Eat which can prevent them  from Various Diseases and  Risks of getting sick.

• It is useful for peoples  to avoid the negative health risks to, diet needs to be nutritional and diverse,Small changes in diet can make an immense difference  to everyones  health.

• Junk food can result in long-term damage, going for unhealthy food stuffs like French fries, pizza, pastries and candy can increase your risk of developing depression, obesity, heart disease and cancer which may also cause to death.

• Eating healthy foods is the cornerstone of good health status and also lower your risk for developing health problems, so this model is very useful and beneficial  for peoples to choose good foods and prevent from getting sick from diseases.

# APPLICATION








# FUTURE SCOPE







# CONCLUSION

• In this project our model is trying to detect food detection and then showing it on viewfinder, live as to whether Helmet is worn or not as we have specified in Roboflow.

• The model tries to solve the problem of severe head injuries that occur due to accidents and thus protects a person’s life.

• The model is efficient and highly accurate and hence reduces the workforce required.

# Refrences

1]Roboflow :- https://roboflow.com/

2] Google images
