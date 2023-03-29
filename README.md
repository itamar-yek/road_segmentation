# road_segmentation

in order to start the algorithm, enter the desired camera feed into the main file and run it.

this algorithm is designed for dual lense camera with resolution of 1280x480, we only use the left lens so the algorithm works for frame size of 640x480.

if your camera has different propreties you can add a resize command or change the default values in the helper file.

our segmentation algorithm assumes that the road is paved and we desire to distinguish between the paved road and the unpaved surrounding.

our algorithm was tested on few frames(30) from the https://lapix.ufsc.br/pesquisas/projeto-veiculo-autonomo/datasets/?lang=en dataset.
we achived 80% accuracy in terms of misclassification rate.

https://user-images.githubusercontent.com/82041015/228685661-3f9eb9f0-5002-4951-b55f-98284c75e89c.mp4

