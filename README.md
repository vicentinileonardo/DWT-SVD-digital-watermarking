# DWT-SVD-digital-watermarking
Repository related to the Catch the Mark competition of the Multimedia Data Security course of University of Trento, academic year 2022/2023
<br><br>
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
+ `utilities folder` - contains original and watermaked images used during the challenge, a `.csv` file used to compute the WPSNR metric and the watermark assigned to our group and 
+ `sample_images` - contains sample images used to compute the ROC curves
+ `test_images` - contains images used to test the detection algorithms
+ `attacked_images` - contains attacked images starting from other groups' watermarked images
+ `howimetyourmark.pdf` - contains the presentation of the group with the final outcomes

# References
+ [1] - [Murty, Pattisapu & Kumar P, Rajesh. (2013). Towards Robust Reference Image Watermarking using DWT- SVD and Edge Detection. International Journal of Computer Applications. 68. 10-15. 10.5120/11606-6975.](https://www.researchgate.net/publication/304201204_Towards_Robust_Reference_Image_Watermarking_using_DWT-_SVD_and_Edge_Detection)
+ [2] - [Alzahrani, A. (2022). Enhanced Invisibility and Robustness of Digital Image Watermarking Based on DWT-SVD. Applied Bionics and Biomechanics, 2022, 5271600. doi:10.1155/2022/5271600](https://www.hindawi.com/journals/abb/2022/5271600/)
