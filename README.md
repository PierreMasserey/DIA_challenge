# DIA_challenge
## Initial data
- 100 images
- Ground truth
- ![image](https://github.com/PierreMasserey/DIA_challenge/assets/43469697/1680abcd-20d5-4487-a576-5221d9cfcc01)
## Workload

## Data preprocessing

## Algorithms
### Machine learning
- Learn with 80 images
- Test with ~20 images

## Tests mesures
### Precision

### Recall
### F-mesure
### Intersection Over Union (IOU)
The IoU calculation involves finding the proportion of the intersection of two regions (the predicted region and the ground truth region) relative to their union. Formally, IoU is calculated as follows:
IoU = Area of Intersection / Area of Union
In the context of object detection, suppose you have a predicted bounding box for an object and a ground truth bounding box for the same object. IoU measures the overlap between these two bounding boxes. A higher IoU value, closer to 1, indicates a better match between the prediction and ground truth, while an IoU close to 0 indicates very little or no overlap between the two boxes.
#### Micro IOU (Only with classes)
#### Macro IOU (General page)

