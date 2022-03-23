# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
	- training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```
The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.
```
You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.


### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
The main idea of this project is application of Neural network for image detection and understanding the different aspects associated with tuning the hyperparameters etc. Object detection or perception is the first step in Self driving cars and its a really important step since without detecting the objects accurately any downstream functionality(path planning, controls) will not perform optimally. 

### Set up
Since I did not have a GPU enabled machine. I used the workspace in which all of the setup was already done.

### Dataset
#### Dataset analysis
![image](https://user-images.githubusercontent.com/102181055/159800622-56c07b8e-e3c2-49a4-8fdd-631dc8bbea3c.png)
![image](https://user-images.githubusercontent.com/102181055/159800656-a9e7616f-f9a2-4252-b14a-aafc1cdef46f.png)
![image](https://user-images.githubusercontent.com/102181055/159800698-5af5e520-7010-48ff-9052-3305c896af58.png)
The dataset contains images related to cars, pedestrians and cyclists. However when doing the explorations on the data randomly I hardly found a cyclist. Also I could see that most of the data was collected in daylight and was missing dark images. There were some images which had small bounding boxes drawn on far away objects which were hardly visible. In some cases a large bounding box was drawn on a group of pedestrians which might lead to some poor training results.

![image](https://user-images.githubusercontent.com/102181055/159801182-9e34fa73-f7ce-4172-b034-049721b7c5a9.png)
The distribution of the data on 2000 images shows very few cyclists hence the model trained on this dataset will perform poorly on cyclist detection. 


#### Cross validation
The data consists of 97 tfrecords(excluding the 3 for test). The data split was done on a random basis to not have an imbalance for different classes. For the first iteration of training I had divided the dataset between test, validation and train and realized that I need more data for training since its a small sample set. Hence I re-grouped using 85/15 split between training and validation.

### Training
#### Reference experiment
![image](https://user-images.githubusercontent.com/102181055/159801914-3d19db60-ff31-4f5a-b2c2-cd46fa90982c.png)
![image](https://user-images.githubusercontent.com/102181055/159801947-1a09a380-9476-4316-905f-86e3f13412fd.png)
![image](https://user-images.githubusercontent.com/102181055/159801974-05b09356-0804-4e8c-a08c-26c6da050dda.png)

As can be seen, the training loss jumps suddenly and it remains high even at the end of the training. This made me reshuffle the dataset as mentioned above and train again to get a new baseline.

![image](https://user-images.githubusercontent.com/102181055/159802176-a1c1b779-9db2-4efa-9968-92ead9cf7673.png)
![image](https://user-images.githubusercontent.com/102181055/159802191-bc2fc527-6cbc-4603-b2fe-2314fa87438e.png)
![image](https://user-images.githubusercontent.com/102181055/159802210-4dc5adab-e34c-402b-84a9-bc6376c21e9d.png)

The new baseline was much better compared to the previous one. However, the losses still high and the precision/recall also seems low. 

#### Improve on the reference
To improve the model the following things were tried:
Augmentations:
The following augmentations were added
1. RGB to gray
2. Random adjust brightness
3. Random adjust contrast
4. Random image scale
The reason for adding the augmentations was to provide more features for dark and low contrast images and to zoom in on some of the images which are far away (small bounding boxes)

Explore Augmentations
![image](https://user-images.githubusercontent.com/102181055/159802797-e622d005-a0c1-4663-a3de-af3c99ecd727.png)
![image](https://user-images.githubusercontent.com/102181055/159802817-66613682-3405-47de-bb91-855120607a97.png)
![image](https://user-images.githubusercontent.com/102181055/159802871-6be3e383-c217-4513-9d2a-4107710c375d.png)
![image](https://user-images.githubusercontent.com/102181055/159802895-b8290f3f-a995-4755-828e-ef0bff5ddf9c.png)

The other thing which I tweaked was to increase the batch size to 4 and step_size tp 5000. This was done to lower the training loss further as just augmentations were not able to acheive it.

Training results with Augmentation + tuning
![image](https://user-images.githubusercontent.com/102181055/159803088-2f92e496-a066-41a8-ae85-06b1e89051f2.png)
![image](https://user-images.githubusercontent.com/102181055/159803119-2b380e62-d5a0-42dd-ad1d-0e9c15f0af04.png)
![image](https://user-images.githubusercontent.com/102181055/159803156-2cadabb7-2e99-4acf-aae7-411be91883c0.png)
![image](https://user-images.githubusercontent.com/102181055/159803206-9baef866-84bd-4cc1-aae6-59d9a54ce2f0.png)
![image](https://user-images.githubusercontent.com/102181055/159803226-7e71c98b-8217-4098-8f04-d121b27cb5ae.png)
![image](https://user-images.githubusercontent.com/102181055/159803247-2e211194-0636-41bd-aee8-d5016b5c3912.png)

As can be seen the training loss is now lower and also the precision and recall values are higher compared to baseline. In the intial portion the Validation and Training loss diverge which might indicate some overfit in the first few iterations but then validation loss converges closer to training loss.

Some Observations watching the generated animations on the test files:
![image](https://user-images.githubusercontent.com/102181055/159803496-57c3539f-b7f1-4ca6-bbf8-312b72d611e1.png)
1.The model has issues identifying objects at further away ranges
![image](https://user-images.githubusercontent.com/102181055/159803598-aa20ec47-3fca-40d9-ab2f-02b07308619b.png)
2. For objects with good visibility the confidence seems low which might be because of low number of training samples
![image](https://user-images.githubusercontent.com/102181055/159803703-07692170-996f-4c6e-b739-814242329ec2.png)
3. The model does not perform well on detecting objects at night, which might be probably due to low number of night time images.
![image](https://user-images.githubusercontent.com/102181055/159803874-86eca443-8333-48a9-a5b4-bd413e0b4461.png)
4. When there are a lot of crowded cars the model does not perform well. Which might be because of poor bounding boxes or ROI issues with training.

This was a really interesting project and I enjoyed working on it.






