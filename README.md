# DWT-SVD-digital-watermarking
Repository related to the Catch the Mark competition of the Multimedia Data Security course of University of Trento, academic year 2022/2023
Group name: howimetyourmark

Contributors:
+ Leonardo Vicentini - [vicentinileonardo](https://github.com/vicentinileonardo)
+ Matteo Darra - [MatteoDarra](https://github.com/MatteoDarra)
+ Sofia Zanrosso - [sofiazanrosso](https://github.com/sofiazanrosso)
+ Roberta Bonaccorso - [robi00](https://github.com/robi00)


Directory structure:
+ `embedding_howimetyourmark.py` - contains the source code of the embedding algorithm
+ `detection_howimetyourmark.py` - contains the source code of the detection algorithm, used to retrieve and compare the watermark previously embedded
+ `attacks.py` - contains the source code of the attacks used to test the robustness of the watermarking algorithm and to attack other groups' watermarked images
+ `tester.py` - contains a simple tester script to test the embedding and detection algorithms
+ `tester2.py` - contains a more complex tester script to test the embedding and detection algorithms
+ `roc_howimetyourmark` - contains the source code for calculating the ROC curves of the detection algorithm
+ `ROC.png` - contains an example of a ROC curve
+ `howimetyourmark.npy` - contains the watermark (1024 bit length)
+ `csf.csv` - contains generic values used to compute the WPSNR metric
+ `utilities folder` - contains original and watermaked images used during the challenge
+ `sample_images` - contains sample images used to compute the ROC curves
+ `test_images` - contains images used to test the detection algorithms
+ `attacked_images` - contains attacked images starting from other groups' watermarked images
+ `presentation.pdf` - contains the presentation of the group with the final outcomes
