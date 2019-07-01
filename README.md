# StigmaSegmentation

The stigma of a squash flower is important to the pollination and fruiting of the plant. Detecting how and when these plants are pollinated is important to biologists to further their understanding of how bees pollinate squash plants. Honey bees are the primary pollinator of squash plants and thus it is important to track when they interact with the stigma. This project focuses on segmenting the stigma from videos of squash plant flowers.

To do this I have attempted using a moving window and convolutional neural networks to segment the stigma from the flower. The dataset is a large set of still images taken every 30 seconds, where the camera is positioned above the flower. The dataset contains a days worth of images collected in this way for about 250 flowers. A hand labeled dataset of bounding ellipse coordinates was created from this dataset for a single image from 50 different flowers. The 50 labeled images are used as a subset as to speed up computations. These 50 images are split into a training and testing set.

To start, I use a moving window of size 200x200 with a stride of 50 over the entire frame for each of the images in the dataset subset. If the center point of the moving window is inside the bounds of the ellipse then it is considered a positive example that contains the stigma, otherwise it is negative.

![Autoencoder](https://imgur.com/BWRkYgU.jpg)

A convolutional autoencoder is then trained with only the positive images of the stigma. It attempts to minimize the reconstruction error of the image. This encodes the representation of a squash flower stigma. After this has been trained then the decoder layers are taken out and replaced with a fully connected layer followed by a sigmoid layer for classification. It should be noted that the encoder weights are saved from the autoencoder and held fixed while the fully connected layer weights are trained. After the weights have been learned all of the weights are set trainable and the model is trained again with a small learning rate to fine tune all of the weights.

![FullyConnected](https://imgur.com/nDCxAue.jpg)

# Results
More results can be viewed [here](https://imgur.com/a/VRoAtQX).
![Flower 1](https://imgur.com/CP5knFV.jpg)
![Flower 2](https://imgur.com/lj6nGmW.jpg)
![Flower 3](https://imgur.com/3oEnWtT.jpg)
