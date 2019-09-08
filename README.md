# food-net

FoodNet, a deep inception net for PennApps 2019f entry "AI Allergy". Created by machine learning engineer, Ryan Park. 
Top-5 accuracy of 93.75% and top-1 accuracy of 72.00% on Food-404.

## Structure

This network was motivated by Inception ResNet v2 described by Szegedy et. al. in the paper ["Inception-v4, Inception-ResNet 
and the Impact of Residual Connections on Learning"](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14806/14311).
After a global average pooling layer, a simple linear 404-D softmax classifier is added. No dropout or batch normalization
is applied due to the lack of overfitting and generally stable gradients.

## Training

FoodNet was trained on Food-404-- the largest public food image dataset available, also made by Ryan Park --using weights 
transferred from an Inception ResNet v2 trained on ImageNet. The top 30% of the network was made trainable and was fine-tuned 
using stochastic gradient descent with momentum. Data augmentation, including distortions, shears, and flips, was also
implemented to ensure a more robust model.

## Results

After about 1.5 hours of training (1 epoch), FoodNet was able to achieve a top-5 accuarcy of 93.75% and a top-1 accuracy of 
72.00% on Food-404. With more training, FoodNet would have been able to achieve even better accuracy on this dataset.
