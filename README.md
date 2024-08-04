# Implementing LoRA for Fine-Tuning

### 🔍📝👉 To learn more about LoRA and other Parameter Efficient Fine-Tuning techniques, check out this [article](https://medium.com/@h4hastak/implementing-lora-for-fine-tuning-50396a22d13c).


## **What is Fine-Tuning?**

State-of-the-art language models are pre-trained on extensive data, providing a broad understanding of language and world knowledge. This allows them to generalize across various language tasks. These models are often large, with billions or even trillions of parameters, and their size continues to grow. Fine-tuning adapts a pre-trained model to specific tasks by adjusting its parameters for more targeted applications, rather than general language understanding.


## **What is LoRA?**

Formally, if fine-tuning is represented as:

![ft](https://github.com/user-attachments/assets/0a7488cb-e71b-4ce7-bd74-122e44cbee54)

Normal Fine-Tuning performs weight updates (ΔW) across the entire pre-trained weight matrix. This approach adjusts the full set of weights, which can be computationally expensive and memory-intensive.LoRA assumes that the matrix (ΔW) also has a low intrinsic dimension. This means that rather than updating the entire weight matrix, LoRA focuses on a low-rank approximation of the weight changes, which can efficiently adapt the model to new tasks while maintaining a reduced computational and memory footprint.

![lora](https://github.com/user-attachments/assets/6775b9c1-5630-43c9-aa4a-8cba94c9916e)

In LoRA (Low-Rank Adaptation), weight updates (ΔW) are decomposed into two smaller matrices, A and B, with reduced dimensions. This method captures the weight changes efficiently while significantly reducing computational and memory requirements.

![difference](https://github.com/user-attachments/assets/277712cc-315a-4abe-9fd3-4a1f475aef7c)


This illustration provides a comprehensive comparison between Normal fine tuning and LoRA. In LoRA, the weight update matrix ΔW is decomposed into two matrices: A, which compresses the information into a low-dimensional space (r), and B, which reconstructs the information back to the original high-dimensional space (d). Here, r≪d, highlights how LoRA effectively reduces the complexity and resource requirements compared to traditional fine-tuning.

## **Emotions Dataset from HuggingFace**

The dataset classifies sentiments of Twitter (now X) data into 6 categories:

![image](https://github.com/user-attachments/assets/e14d3ebd-264e-4014-a6a1-86f706e790be)


## **Results**

The model was tested on multiple configurations of scale through combinations of rank and alpha.

| Hyperparameter        | Value         |
|-----------------------|---------------|
| **Batch Size**        | 16            |
| **Learning Rate (LR)**| 1e-4          |
| **Token Max Length**  | 64            |
| **Ranks**             | 1, 4, 8, 16   |
| **Scale**             | 1, 2          |

Performance was consistent across configurations, with a slight accuracy variation (about 0.6%). The model achieved a **classification accuracy of 93%** within a single epoch and used a rank-decomposition of 1, **reducing parameter size by 99.99%**.


## **Analyzing Misidentified Classes**

Interesting trends in class prediction distribution:

![image](https://github.com/user-attachments/assets/75479356-ddec-4d08-b48a-a7c1e6224baa)

- The class "**surprise**" has the **highest proportion of misidentified counts** across most configurations, indicating it is the hardest class to predict accurately.
- The classes "**love**" and "**joy**" generally have lower misidentification proportions, indicating these emotions are **easier to predict correctly** across different configurations.

