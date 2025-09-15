# CLNet: Real-Time Pilot Cognitive Load Monitoring
 
Rapid evolution of intelligent flight cockpits necessitates real-time monitoring of pilots’ cognitive load to ensure flight safety. This study introduces CLNet, a lightweight neural network designed for the accurate classification of pilots’ cognitive load states. Utilizing multi-scale spatiotemporal convolution, CLNet enhances feature extraction from a variety of physiological signals, significantly boosting classification accuracy. Ablation studies high-light the critical roles of the squeeze-and-excitation and temporal gate convolution mod-ules in optimizing network performance. To evaluate the effectiveness of CLNet, we devel-oped the Airfield Traffic Pattern Cognitive Load (ATPCL) dataset, which includes electroen-cephalogram (EEG), electrocardiogram (ECG), and electromyography (EMG) signals record-ed during key flight phases. On the ATPCL dataset, CLNet achieved an accuracy of 95.1%. We also developed an integrated online monitoring system for real-time data collection, processing, and visualization. This system employs the CLNet algorithm for rapid cognitive load assessment and updating within milliseconds. Our system provides a valuable tool for real-time pilot cognitive load evaluation, supporting advancements in aviation research and applications.
 

# Introduction
 
The rapid evolution of intelligent flight cockpits necessitates real-time monitoring of pilots’ cognitive load to ensure safety. CLNet (Cognitive Load Network) is a novel, ultra-lightweight neural architecture—just 27 K parameters—designed for accurate, millisecond-scale classification of pilot workload (low, medium, high).
 
 
## CLNet Framework
 
<img width="6675" height="4575" alt="Figure2" src="https://github.com/user-attachments/assets/d1df865c-b850-4842-a5bd-4f0d7799341c" />


 
 
## ATPCL Dataset
 

The Airfield Traffic Pattern Cognitive Load (ATPCL) dataset was recorded on a flight simulator during the cruise, takeoff, and landing phases. It includes:  
- 15 professionally trained cadets from Beihang University  
- Completion of a five-leg flight task sequence  
- Simultaneous EEG, EMG, and ECG signal recordings
  
*Since all participants are civil aviation flight trainees who are about to enter the workforce, we regret that we are unable to make the dataset publicly available due to privacy and safety concerns for the pilots.*


 ---
 
We welcome your feedback and contributions!  
 
