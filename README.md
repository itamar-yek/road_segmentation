# road_segmentation

in order to start the algorithm, enter the desired camera feed into the main file and run it.

this algorithm is designed for dual lense camera with resolution of 1280x480, we only use the left lens so the algorithm works for frame size of 640x480.

if your camera has different propreties you can add a resize command or change the default values in the helper file.

our segmentation algorithm assumes that the road is paved and we desire to distinguish between the paved road and the unpaved surrounding.

our algorithm was tested on few frames (30) from the https://lapix.ufsc.br/pesquisas/projeto-veiculo-autonomo/datasets/?lang=en dataset.
we achived 80% accuracy in terms of misclassification rate.




https://user-images.githubusercontent.com/82041015/228686175-e7015664-5d26-41b5-8931-2e33344aaf9b.mp4



https://technionmail-my.sharepoint.com/:v:/r/personal/itamar_yek_campus_technion_ac_il/Documents/Road%20Segmentation.mp4?csf=1&web=1&e=31DBbS
