# EC601 Medical VQA
Visual Question Answering (VQA) in the medical domain aims to answer a clinical question presented with a medical image. The system could support clinical education, clinical decision, and patient education. For doctors, It helps to interpret complex clinical images and make more accurate clinical decisions. 
For patients, it leads to better understand their health condition.　　

## Our model and method
![image](https://user-images.githubusercontent.com/32304880/146292628-9ac2b9ac-2112-41a3-bbef-914b2b11f49c.png)
- Image Encoder: use VGG19 to catch image features [1, 1024, 1]
- Question Encoder: use LSTM to catch question features [1, 1024, 1]
- Fusion: Concatenate and reshape the two features to [2, 32, 32], use Conv2D to fuse them to [1, 32, 32], then reshape it to [1, 1024, 1]
- Fully Connected: Use fully connected layer to calculate the possibility of each answer.   

## Result
![image](https://user-images.githubusercontent.com/32304880/146292846-b8b5e288-ebd4-4611-bf56-5024b3dd5aba.png)
- The best result is the 10th model with mean and standard deviation based on the dataset, and batch size 1
- Small dataset causes overfitting quickly, since the train accuracy reached almost 100% in the end.
- The model can predict simple open-ended questions accurately, but performs not well on close-ended questions.
- The inaccurate results are caused by lack of questions, since different organs and symptoms can be combined into a lot of close-ended questions. It is hard to train all of them.

[Poster Link](https://github.com/YukoIshikawa/EC601_Medical_VQA/blob/main/EC601_poster.pdf)
## Sprint 1
We defined product mission, MVP and MVP user stories, did comprehensive literature review, and determined technologies for the evaluation and development environment setup.   
[Presentation Link](https://github.com/YukoIshikawa/EC601_Medical_VQA/blob/main/Medical%20_VQA_sprint1.pdf)
## Sprint 2
We duplicated the open source [basic-VQA](https://github.com/tbmoon/basic_vqa), tested the model and analyzed the results.    
[Presentation Link](https://github.com/zhaojun-szh-9815/EC601/blob/main/VQA_Project/Sprint_2.md)
## Sprint 3  
- We duplicated the other open source [Mixture of Enhanced Visual Features (MEVF)](https://github.com/aioz-ai/MICCAI19-MedVQA) and test the model and analyzed the results
- We did dataset analysis that we use for our model for the presentation   
[MEVF Model Test and Analysis Link](https://github.com/YukoIshikawa/EC601_Medical_VQA/tree/main/Med-VQA)  
[Presentation Link](https://github.com/RuilingZ/EC601-1/blob/main/VQA_Project/sprint%203.md)

## Sprint 4
- We generated a bigger dateset, trained and tested our model on it and analyze the results.   
- We did more dataset analysis to consider how to improved our model and tested our model fixed based on the analysis.   
[Dataset Link](https://github.com/YukoIshikawa/EC601_Medical_VQA/tree/main/Gen-Dataset)  
[Presentation Link](https://github.com/zhaojun-szh-9815/EC601/blob/main/VQA_Project/Sprint_4/README.md)

