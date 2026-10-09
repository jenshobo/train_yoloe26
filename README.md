Programs used to annotate and train Yoloe26 image recognition models.

## How to use

For this program to work you'll first need a dataset, this is basically just a massive number of images of the subject you wish to detect.

### Annotation

If you have not done so yet, you can automatically annotate the images in your dataset using the ```annotate.py``` file, however this does use basic image recognition using a generic Yoloe26 model. So the result might not be accurate. The benefit however is time saving and you can use the x model of Yoloe26 to increase accuracy as you'll only need to run this ones.

### Datasets

Datasets contain a yaml pointing to each folder used in the dataset and which classes (names to recognise) the model should contain and is defined in your dataset. This yaml should look roughly like this:
```
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['dogs', 'cats']
```
Here the dataset and resulting model use the classes ```dogs``` and ```cats```.

It is up to you to determine the ratio of images in each folder, although it is recommended to use about 70% for training and the remaining 30% for validation and testing. Your free to change these percentages if you think it is best to do so.

### Training

To train the model, make sure the pointers in ```train.py``` point to your used dataset and run it. It takes a while to train so come back a little while later. Ones done the program should have created a ```.pt``` file a new folder called ```runs/segment/train/weights```. These files are your results and can be used in other projects to detect the classes you specified.
