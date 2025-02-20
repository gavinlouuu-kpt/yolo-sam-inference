This pipeline will take a path as input (containing images .png, .tiff, .jpg etc.) and return the following:
1. Through yolo will return A list of bounding boxes (essentially identifying background and images with target)
2. The bounding box will be used as box prompt in sam (the bounding box will be paired with image)

The output of this package will give:
1. Segmented mask of all the images in a folder
2. Original image cropped by bounding box
3. Metrics including the following: Mask area, mask circularity, mask deformability, convex hull of the mask, distribution of pixel intensity within the mask area from the original image
