# ArtExtract: Task 1 (Multi-Task Classification) & Task 2 (Painting Similarity)
![thumbnail](assets/thumbnail.jpg)
Find notebooks at:
- Task 1 : 
    - notebook: [click here](task1-CRNN-WikiArt-Classification.ipynb) 
    - pdf: [click here](pdf/task1-CRNN-WikiArt-Classification.pdf)
- Task 2 : 
    - notebook: [click here](task2-Painting-Similarity.ipynb) 
    - pdf: [click here](pdf/task2-Painting-Similarity.pdf)

The methodologies/approach for is explained inside the corresponding notebooks.

Run following to install dependencies:
```bash
pip install -r requirements.txt
```

### Results:
---
#### Task 1: Multi-Task Classification 
 
(please refer [notebook](task1-CRNN-WikiArt-Classification.ipynb) for explanation)
| Architecture | Style (Top-1 / F1) | Artist (Top-1 / F1) | Genre (Top-1 / F1) | Global F1 |
| :--- | :---: | :---: | :---: | :---: |
| ResNet18 (10e Frozen) | 46.48% / 0.3851 | 71.03% / 0.6872 | 71.08% / 0.6575 | 0.5766 |
| ResNet50 (10e Frozen) | 54.65% / 0.4799 | 79.21% / 0.7746 | 74.96% / 0.7096 | 0.6547 |
| **ResNet50 (10e Frozen + 10e FT)** | **59.33%** / **0.5502** | **83.25%** / **0.8179** | **77.06%** / **0.7401** | **0.7027** |

*\*Note: The RNN and Multi-Head configurations remained constant across all experiments. (10e = 10 epochs, FT = Fine-Tuned).*

#### Task 2: Painting Similarity
![potrait](assets/sim_vis-1.png)

![canvas](assets/sim_vis-2.png)

![mother-and-baby](assets/sim_vis-3.png)
