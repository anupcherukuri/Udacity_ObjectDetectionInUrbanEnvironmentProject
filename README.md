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
