# Implementing LoRA for fine-tuning

To learn more about Fine-Tuning and LoRA, check out this [article](https://medium.com/@h4hastak/implementing-lora-for-fine-tuning-50396a22d13c).


**What is Fine-Tuning?**

State-of-the-art language models are pre-trained on extensive data, providing a broad understanding of language and world knowledge. This allows them to generalize across various language tasks. These models are often large, with billions or even trillions of parameters, and their size continues to grow.
Fine-tuning adapts a pre-trained model to specific tasks by adjusting its parameters for more targeted applications, rather than general language understanding.


**What is LoRA?**

Formally, if fine-tuning is represented as: 
![image](https://github.com/user-attachments/assets/514f5b40-c2bc-424c-8af3-b7c876c232dd)

[LoRA](https://arxiv.org/abs/2106.09685)  assumes that the matrix Δ(W) also has a low intrinsic dimension. Instead of updating the entire weight matrix, LoRA focuses on a low-rank approximation of weight changes, enabling efficient adaptation to new tasks while reducing computational and memory requirements.

**Emotions Dataset from HuggingFace**

The dataset classifies sentiments of Twitter (now X) data into 6 categories:
![image](https://github.com/user-attachments/assets/e14d3ebd-264e-4014-a6a1-86f706e790be)


**Results**
The model was tested with a batch size of 16, a learning rate of 1e-4, and ranks of 1, 4, 8, and 16 with scales of 1 and 2 for each. Performance was consistent across configurations, with a slight accuracy variation (about 0.6%). The model achieved a **classification accuracy of 93%** within a single epoch and used a rank-decomposition of 1, **reducing parameter size by 99.99%**.

**Analysing Misidentified classes**

Interesting trends in class prediction distribution: 
![image](https://github.com/user-attachments/assets/e7c8e885-0225-4ac0-bf84-3e977d18b257)

	• The class "surprise" has the highest proportion of misidentified counts across most configurations, indicating it is the hardest class to predict accurately.
	• The classes "love" and "joy" generally have lower misidentification proportions, indicating these emotions are easier to predict correctly across different configurations.
