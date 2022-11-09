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
+ `roc_howimetyourmark.py` - contains the source code for calculating the ROC curves of the detection algorithm
<br><br>
+ `ROC.png` - contains an example of a ROC curve
+ `csf.csv` - contains generic values 
+ `utilities folder` - contains original and watermaked images used during the challenge, a `.csv` file used to compute the WPSNR metric and the watermark assigned to our group and 
+ `sample_images` - contains sample images used to compute the ROC curves
+ `test_images` - contains images used to test the detection algorithms
+ `attacked_images` - contains attacked images starting from other groups' watermarked images
+ `howimetyourmark.pdf` - contains the presentation of the group with the final outcomes

