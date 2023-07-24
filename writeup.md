Self-Driving Car Beta Testing Nanodegree 

This is a template submission for the midterm second course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : 3D Object Detection (Midterm). 


## 3D Object detection

We have used the [Waymo Open Dataset's](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) real-world data and used 3d point cloud for lidar based object detection. 

- Configured the ranges channel to 8 bit and view the range /intensity image (ID_S1_EX1)
- Used the Open3D library to display the lidar point cloud on a 3d viewer and identifying 10 images from point cloud.(ID_S1_EX2)
- Created Birds Eye View perspective (BEV) of the point cloud,assign lidar intensity values to BEV,normalize the heightmap of each BEV (ID_S2_EX1,ID_S2_EX2,ID_S2_EX3)
- In addition to YOLO, use the [repository](https://review.udacity.com/github.com/maudzung/SFA3D) and add parameters ,instantiate fpn resnet model(ID_S3_EX1)
- Converted BEV coordinates into pixel coordinates and convert model output to bounding box format  (ID_S3_EX2)
- Computed intersection over union, assign detected objects to label if IOU exceeds threshold (ID_S4_EX1)
- Computed false positives and false negatives, precision and recall(ID_S4_EX2,ID_S4_EX3)


The project can be run by running 

```
python loop_over_dataset.py
```
All training/inference is done on GTX 2060 in windows 10 machine.


## Step-1: Compute Lidar point cloud from Range Image

In this we are first previewing the range image and convert range and intensity channels to 8 bit format. After that, we use the openCV library to stack the range and intensity channel vertically to visualize the image.

- Convert "range" channel to 8 bit
- Convert "intensity" channel to 8 bit
- Crop range image to +/- 90 degrees  left and right of forward facing x axis
- Stack up range and intensity channels vertically in openCV

For the next part, we use the Open3D library to display the lidar point cloud on a 3D viewer and identify 10 images from point cloud
- Visualize the point cloud in Open3D
- 10 examples from point cloud  with varying degrees of visibility

The range image sample

<img width="1317" alt="Screenshot 2023-07-24 at 12 47 46" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/42324d5f-bf67-4752-9974-572b399bd1b2">


Point cloud images:

<img width="1178" alt="Screenshot 2023-07-24 at 12 52 04" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/81bb8968-95b5-4d11-b224-52dc2e23ebe9">

<img width="1222" alt="Screenshot 2023-07-24 at 12 52 11" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/fef92241-026c-4eea-8acd-98c7aaab8a4b">

<img width="1237" alt="Screenshot 2023-07-24 at 12 52 17" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/b97224ba-fdb0-4569-b345-3483935a82f9">

<img width="787" alt="Screenshot 2023-07-24 at 12 52 25" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/33a1fffe-7143-4b26-a8e8-271f5855a1c9">

<img width="815" alt="Screenshot 2023-07-24 at 12 52 32" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/290c443f-14e8-423a-b142-b74ad1f1ef64">

<img width="1172" alt="Screenshot 2023-07-24 at 12 52 39" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/1134a775-5e49-476c-b962-0ae5cc0ef3e0">

<img width="1217" alt="Screenshot 2023-07-24 at 12 52 44" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/35c69cb9-cc21-46a8-a609-80e11a0caff0">


Stable features include the tail lights, the rear bumper  majorly. In some cases the additional features include the headover lights, car front lights, rear window shields. These are identified through the intensity channels . The chassis of the car is the most prominent identifiable feature from the lidar perspective. The images are analysed with different settings and the rear lights are the major stable components, also the bounding boxes are correctly assigned to the cars (used from Step-3).


## Step-2: Creaate BEV from Lidar PCL

In this case, we are:
- Converting the coordinates to pixel values
- Assigning lidar intensity values to the birds eye view BEV mapping
- Using sorted and pruned point cloud lidar from the  previous task
- Normalizing the height map in the BEV
- Compute and map the intensity values


## Step-3: Model Based Object Detection in BEV Image

Here we are using the cloned [repo](https://github.com/maudzung/SFA3D) ,particularly the test.py file  and extracting the relevant configurations from 'parse_test_configs()'  and added them in the 'load_configs_model' config structure.

- Instantiating the fpn resnet model from the cloned repository configs
- Extracting 3d bounding boxes from the responses
- Transforming the pixel to vehicle coordinates
- Model output tuned to the bounding box format [class-id, x, y, z, h, w, l, yaw]


As the model input is a three-channel BEV map, the detected objects will be returned with coordinates and properties in the BEV coordinate space. Thus, before the detections can move along in the processing pipeline, they need to be converted into metric coordinates in vehicle space.

A sample preview of the bounding box images:

<img width="665" alt="Screenshot 2023-07-24 at 12 54 59" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/94d143de-7c93-4c31-b275-8a8072b22d00"><img width="658" alt="Screenshot 2023-07-24 at 12 54 55" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/d1c7fd3a-2e28-41b8-882f-da9ab8f6364d">


## Step-4: Performance detection for 3D Object Detection

In this step, the performance is computed by getting the IOU  between labels and detections to get the false positive and false negative values.The task is to compute the geometric overlap between the bounding boxes of labels and the detected objects:

- Assigning a detected object to a label if IOU exceeds threshold
- Computing the degree of geometric overlap
- For multiple matches objects/detections pair with maximum IOU are kept
- Computing the false negative and false positive values
- Computing precision and recall over the false positive and false negative values


The precision recall curve is plotted showing similar results of precision =0.996 and recall=0.81372

<img width="675" alt="Screenshot 2023-07-24 at 13 02 55" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/d6407788-85f3-43d4-b693-7ef839d2ba8c">


In the next step, we set the 
```python
configs_det.use_labels_as_objects=True
```
 which results in precision and recall values as 1.This is shown in the following image:
 
<img width="934" alt="Screenshot 2023-07-24 at 13 03 16" src="https://github.com/Vvlladd/midTermProject_3DObjectDetection/assets/88333833/8f052ee2-5ff4-4642-a773-e34f2917b1a0">


## Summary of Lidar based 3D Object Detection

The project highlights the importance of using lidar for achieving stable tracking. The conversion of range data into point clouds using spatial volumes or points (or CNN networks) plays a crucial role in subsequent analysis. To enable 3D object detection, employing resnet/darknet and YOLO is essential for converting these high-dimensional point cloud representations into object detections with bounding boxes. Evaluating the performance through maximal IOU mapping, mAP, and representing the precision/recall of the bounding boxes are vital steps to comprehend the effectiveness of lidar-based detection.
