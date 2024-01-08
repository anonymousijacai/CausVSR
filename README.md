# CausVSR: Causality Inspired Visual Sentiment Recognition

## Introduction
Visual Sentiment Recognition (VSR) is an emerging field that aims to detect emotional tendencies within visual content. Despite its growing significance, detecting emotions depicted in visual content, such as images, faces challenges, notably the emergence of misleading or spurious correlations of the contextual information. In response to these challenges, we propose a causality inspired VSR approach, called CausVSR. CausVSR is rooted in the fundamental principles of Emotional Causality theory, mimicking the human process from receiving emotional stimuli to obtaining emotional states. CausVSR takes a deliberate stride toward conquering the hurdles encountered in VSR. It harnesses the power of a structural causal model, intricately designed to encapsulate the dynamic causal interplay between input visual contents, such as images, and their corresponding pseudo sentiment regions. This strategic approach allows for a deep exploration of contextual information, elevating the accuracy of emotional inference. Additionally, CausVSR utilizes a global category elicitation module, strategically employed to execute front-door adjustment techniques,  effectively detecting and handling spurious correlations. Experiments, conducted on five widely-used datasets, demonstrate CauVSR's superiority in enhancing emotion perception within VSR, surpassing existing methods. 

## Architecture
![](https://github.com/anonymousijacai/CausVSR/blob/main/introduction.jpg)

## Dependencies
- <code>python</code> (tested on python3.7)
- <code>PyTorch</code>  (tested on 1.2.0)
- <code>torchvision</code>  (tested on 0.4.0)

## Installiation
 1. Clone this repository.
 2. <code>pip install -r requirements.txt</code>

## Data Preparation
 1. Download large-scale dataset FI-8 [here](https://drive.google.com/drive/folders/1gz5WhybpFT7F3YJ8Hl-6gxYWq12Gmbax?usp=drive_link), and put the splited dataset into <code>CauVSR/FI</code>.
 2. Download small-scale dataset [Emotion-6](http://chenlab.ece.cornell.edu/downloads.html).


## Train
1. Launch training by the command below:
   ```
   $ python main.py
   ```
   
## Visualization
- The Causal Psuedo Sentiment Maps can be download [here](https://drive.google.com/drive/folders/1Q4MLwrv5lJamNgAGeYL1dg-4JdmpLoHd?usp=drive_link).

## TODO (_if be accepted_)
- Release the code of the Global Category Elicitation Module (GCSM).
- Release the code of the Surface Normal Loss (L<sub>SNL</sub>).
- Release the code of drawing confusion matrix.
- All training weights of experiments will be available after acceptance of the paper (the training weight on FI-8 dataset can be obtained [here](https://drive.google.com/file/d/1dZD9dfyB104KgRxUz2NDhkaEPMq2C9MK/view?usp=drive_link)).
- All training models will be available after acceptance of the paper.

## References
Our code is developed based on:
- [WSCNet: Weakly Supervised Coupled Networks for Visual Sentiment Classification and Detection.](https://ieeexplore.ieee.org/document/8825564)
- [DCNet: Weakly Supervised Saliency Guided Dual Coding Network for Visual Sentiment Recognition.](https://www.researchgate.net/publication/374300197_DCNet_Weakly_Supervised_Saliency_Guided_Dual_Coding_Network_for_Visual_Sentiment_Recognition)

Thanks for their great work!

