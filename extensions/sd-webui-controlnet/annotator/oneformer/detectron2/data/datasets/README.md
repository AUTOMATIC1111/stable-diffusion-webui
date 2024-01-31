

### Common Datasets

The dataset implemented here do not need to load the data into the final format.
It should provide the minimal data structure needed to use the dataset, so it can be very efficient.

For example, for an image dataset, just provide the file names and labels, but don't read the images.
Let the downstream decide how to read.
