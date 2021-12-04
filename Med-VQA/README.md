# Architecture 
<img src="https://user-images.githubusercontent.com/32304880/144708257-b499d16f-09c3-417c-97cf-74d0b19f90a1.png" width="500px">

## Image feature extraction
initialized by pretrained weights from MAML and CDAE
- MAML - Model-Agnostic Meta-Learning
- CDAE - Convolutional Denoising Auto-Encoder 

# Dataset 
VQA-RAD: includes 315 medical images and 3515 correspoding questions (3064 for training and 451 for testing).     

# Duplicated Test Result 
The results show that the accuracy on close-ended questions are higher than on the open-ended questions since open-ended questions ask about the detail description and require longer answers so they need more information from input images. But it is still difficult to provide enough information from the proposed image feature extraction.   
- Open-ended  
VQA Accuracy: 39.7    
<img src="https://user-images.githubusercontent.com/32304880/144707915-2019a650-1ee5-4578-9c29-c5ab1ca9fa91.png" width="400px"> <img src="https://user-images.githubusercontent.com/32304880/144707929-702a83a0-6639-4894-8226-dac78856a0ee.png" width="400px">

- Close-ended   
VQA Accuracy: 73.2    
<img src="https://user-images.githubusercontent.com/32304880/144707874-2ce59256-9bd8-4157-a14a-d4c15aea9097.png" width="400px"> <img src="https://user-images.githubusercontent.com/32304880/144707886-69c894bb-0071-42cb-a792-8fec2d8d20c3.png" width="400px">


# Summary
- The model performance for closed-ended questions is better than our model since they use bilinear attention networks
- It requires more works to understand the framework and think of the method to improve the model 
- We are going to stick to our model we have now and consider other method to improve our model  
