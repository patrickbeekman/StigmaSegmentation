# StigmaSegmentation

The stigma of a squash flower is important to the pollination and fruiting of the plant. Detecting how and when these plants are pollinated is important to biologists to further their understanding of how bees pollinate squash plants. Honey bees are the primary pollinator of squash plants and thus it is important to track when they interact with the stigma. This project focuses on segmenting the stigma from videos of squash plant flowers.

To do this I have attempted using a moving window and convolutional neural networks to segment the stigma from the flower. The dataset is a large set of still images taken every 30 seconds, where the camera is positioned above the flower. The dataset contains a days worth of images collected in this way for about 250 flowers. A hand labeled dataset of bounding ellipse coordinates was created from this dataset for a single image from 50 different flowers. The 50 labeled images are used as a subset as to speed up computations.

To start, I use a moving window of size 200x200 with a stride of 50 over the entire frame for each of the images in the dataset subset. If the center point of the moving window is inside the bounds of the ellipse then it is considered a positive example that contains the stigma, otherwise it is negative. 
