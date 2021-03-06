# PR-OCT-Neural-Net-Source-Code
The source code for Zeng, Yifeng, Shiqi Xu, et al. "Real-time colorectal cancer diagnosis using PR-OCT with deep learning." Theranostics 10.6 (2020): 2587. Please cite our publication if you find this work is helpful!

![Demo](https://github.com/ZenithZyf/PR-OCT-Neural-Net-Source-Code/blob/main/PROCT_Normal_Cancer.gif)

This is implemented using ```Pytorch```.

## Required packages:

Install the required packages:

```
apt-get install tk-dev python-tk
pip install pandas
pip install opencv-python
pip install requests
```

## Training:

The network can be trained by running train.py. We are using a CSV dataloader which can read in annotations in CSV format. To train a network, use below command:

```
python train.py --dataset csv --csv_train train_labels.csv  --csv_classes octID.csv  --csv_val val_labels.csv
```

The csv_train command will load the training data, and the csv_classes will load the image classes. You can find two example dataset in the repo: "train_labels.csv" and "octID.csv". Note that csv_val command is optional, if discarded, there will be no validataion. Again, a validation example dataset is also in the repo: "val_labels.csv".

One example of our training annotation is:

![Demo](https://github.com/ZenithZyf/PR-OCT-Neural-Net-Source-Code/blob/main/exampleImages/train_example.png)

*Note that this image is cropped for better visualization.*

## Pre-trained model:

The pre-trained model can be download from [Optical and Ultrasound Imaging Lab](https://opticalultrasoundimaging.wustl.edu/).

## Visualization:

To visualize the object detection result, one can use 'visualize.py':

```
python visualize.py --dataset csv --csv_classes octID.csv --csv_val val_labels.csv --model model_final.pt
```

This will visualize bounding boxes on the validation set. One example of our visualizing output on a test image is:

![Demo](https://github.com/ZenithZyf/PR-OCT-Neural-Net-Source-Code/blob/main/exampleImages/test_example.png)

*Note that this image is cropped for better visualization.*

## Annotation and Class Mapping:

The annotation example can be found in "train_labels.csv". Specifically, the expected format for each line is:

```
path/to/image.bmp, x1, y1, x2, y2, class_name
```

Some images may not contain any image class, you can add them as follow:

```
path/to/image.bmp, , , , , 
```

The class mapping is from class name to a specific ID. One can find an example in "octID.csv". Specifically, the format shall be:

```
class_name, id
```

## Annotation tool:

We used [tzutalin's labelImg toolkit](https://github.com/tzutalin/labelImg) to generate the csv file with annotation.

## Acknowledgements

Most of our source codes are borrowed from the [keras retinanet implementation](https://github.com/fizyr/keras-retinanet), [yhenon's retinanet implementation](https://github.com/yhenon/pytorch-retinanet) and adapted to our specific applications.
