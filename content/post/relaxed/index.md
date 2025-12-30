---
title: "Effect of equivariance on training dynamics"
date: "2024-09-01"
summary: "🔍 Searching for the Best Framework for GDL+TDL Methods? Look no further! This blog post reveals how the TopoX suite boosts modularity and optimizes time and memory usage for methods like Equivariant Simplicial Complexes 🚀"
tags:
  - Demo
  - External
draft: false
image:
  placement: 2

---

<meta http-equiv="refresh" content="0;url=https://gram-blogposts.github.io/blog/2024/relaxed-equivariance/">
<script>window.location.replace("https://gram-blogposts.github.io/blog/2024/relaxed-equivariance/");</script>

Si no se redirige automáticamente, haz clic en este enlace: <a href="https://gram-blogposts.github.io/blog/2024/relaxed-equivariance/">Ir al post externo</a>.








Effect of equivariance on training dynamics
Can relaxing equivariance help in finding better minima?

Authors
Affiliations
Diego Canez

University of Amsterdam

Nesta Midavaine

University of Amsterdam

Thijs Stessen

University of Amsterdam

Jiapeng Fan

University of Amsterdam

Sebastian Arias

University of Amsterdam

Alejandro Garcia (supervisor)

University of Amsterdam

Published
July 20, 2024

Contents
Background
Regular GCNN
Steerable GCNN
Relaxed regular GCNN
Relaxed steerable GCNN
Methodology
Datasets
Training Dynamics Evaluation
Results
Smoke Plume with Full Equivariance
Super Resolution
Concluding Remarks
References
Group Equivariant Convolutional Network (G-CNN) has gained significant traction in recent years owing to their ability to generalize the property of CNNs being equivariant to translations in convolutional layers. With equivariance, the network is able to exploit groups of symmetries and a direct consequence of this is that it generally needs less data to perform well. However, incorporating such knowledge into the network may not always be advantageous, especially when the data itself does not exhibit full equivariance. To address this issue, the G-CNN was modified, introducing relaxed group equivariant CNNs (RG-CNN). Such modified networks adaptively learn the degree of equivariance imposed on the network, i.e. enabling it to operate on a level between full equivariance and no equivariance. Surprisingly, for rotational symmetries on fully equivariant data, 
[1]
 found that a G-CNN exhibits poorer performance compared to a RG-CNN. This is a surprising result because a G-CNN, i.e. a fully equivariant network, is designed to perform well on fully equivariant data. Possibly the training dynamics benefit from relaxing of the equivariance constraint. To investigate this, we use the framework described in 
[2]
 for measuring convexity and flatness using the Hessian spectra.

Inspired by the aforementioned observations, this blog post aims to answer the question: How does the equivariance imposed on a network affect its training dynamics? We identify the following subquestions:

How does equivariance imposed on a network influence generalization?
How does equivariance imposed on a network influence the convexity of the loss landscape?
We tackle these subquestions by analyzing trained models to investigate their training dynamics.

In view of space constraint, in this blogpost, we omit our reproducibility study and refer the readers to our extended blog post. Nevertheless, our reproducibility studies corroborated the following claims:

Relaxed steerable G-CNN outperforms steerable G-CNN (fully equivariant network) on fully rotationally equivariant data as shown in the experiment on the super resolution dataset in 
[1]
.
Relaxed G-CNN outperforms G-CNN on non-fully rotationally equivariant data as shown in the experiment on the smoke plume dataset in 
[3]
.
Background
Regular G-CNN
Consider the segmentation task depicted in the picture below.

Figure 1
Annotated segmented image taken from 
[4]

Naturally, applying segmentation on a rotated 2D image should give the same segmented image as applying such rotation after segmentation. Mathematically, for a neural network 
N
N
NN to be equivariant w.r.t. the group 
(
G
,
⋅
)
(G,⋅), such as 2D rotations, then the following property needs to be satisfied:

 
To build such a network, it is sufficient that each of its layers is equivariant in the same sense. Recall that a CNN achieves equivariance to translations by sharing weights in kernels that are translated across the input in each of its convolution layers. Hence, a G-CNN extends this concept of weight sharing to achieve equivariance w.r.t an arbitrary locally-compact group 
G
G.

For now on we will focus on affine groups, i.e., let 
G
:
=
Z
n
⋊
H
G:=Z 
n
 ⋊H, where 
H
H can be, for example, the rotation subgroup 
S
O
(
n
)
SO(n) and 
Z
n
Z 
n
 , the discrete translation group.

Furthermore, we’ll consider an input signal of 
c
0
c 
0
​
  channels on an 
n
n-dimensional grid 
f
0
:
Z
n
→
R
c
0
f 
0
​
 :Z 
n
 →R 
c 
0
​
 
 , e.g. RGB images (
f
:
Z
2
→
R
3
f:Z 
2
 →R 
3
  ).

Lifting convolution
The first layer of a G-CNN lifts the input signal 
f
0
f 
0
​
  to the group 
G
G using the kernel 
ψ
:
Z
n
→
R
c
1
×
c
0
ψ:Z 
n
 →R 
c 
1
​
 ×c 
0
​
 
  as follows:

 
where 
x
∈
Z
n
x∈Z 
n
  and 
h
∈
H
h∈H. This yields 
f
1
:
Z
n
×
H
→
R
c
1
f 
1
​
 :Z 
n
 ×H→R 
c 
1
​
 
  which is fed to the next layer.

G
G-equivariant convolution
Then, 
f
1
f 
1
​
  undergoes 
G
G-equivariant convolution with a kernel 
Ψ
:
G
→
R
c
2
×
c
1
Ψ:G→R 
c 
2
​
 ×c 
1
​
 
 :

  
 
where 
x
∈
Z
n
x∈Z 
n
  and 
h
∈
H
h∈H. This outputs the signal 
f
2
:
Z
n
×
H
→
R
c
2
f 
2
​
 :Z 
n
 ×H→R 
c 
2
​
 
 . This way of convolving is repeated for all subsequent layers until the final aggregation layer, e.g. linear layer, if there is one.

Note that for regular group convolution to be practically feasible, 
G
G has to be finite or addecuatly subsampled. Some of these limitations can be solved by steerable group convolutions.

Steerable G-CNN
First, consider the group representations 
ρ
i
n
:
H
→
R
c
in
×
c
in
ρ 
in
​
 :H→R 
c 
in
​
 ×c 
in
​
 
  and 
ρ
o
u
t
:
H
→
R
c
out
×
c
out
ρ 
out
​
 :H→R 
c 
out
​
 ×c 
out
​
 
 . To address the aforementioned equivariance problem, 
G
G-steerable convolution modifies 
G
G-equivariant convolution with the following three changes:

The input signal becomes 
f
:
Z
n
→
R
c
in
f:Z 
n
 →R 
c 
in
​
 
 .
The kernel 
ψ
:
Z
n
→
R
c
out
×
c
in
ψ:Z 
n
 →R 
c 
out
​
 ×c 
in
​
 
  used must satisfy the following constraint for all 
h
∈
H
h∈H: 
Standard convolution only over 
Z
n
Z 
n
  and not 
G
:
=
Z
n
⋊
H
G:=Z 
n
 ⋊H is performed.
To secure kernel 
ψ
ψ has the mentioned property, we precompute a set of non-learnable basis kernels 
(
ψ
l
)
l
=
1
L
(ψ 
l
​
 ) 
l=1
L
​
  which do have it, and define all other kernels as weighted combinations of the basis kernels, using learnable weights with the same shape as the kernels.

Therefore, the convolution is of the form:

 
 
Whenever both 
ρ
i
n
ρ 
in
​
  and 
ρ
o
u
t
ρ 
out
​
  can be decomposed into smaller building blocks called irreducible representations, equivariance w.r.t. infinite group 
G
G is achieved (see Appendix A.1 of 
[5]
).

Relaxed G-CNN
The desirability of equivariance in a network depends on the amount of equivariance possessed by the data of interest. To this end, relaxed G-CNN is built on top of a regular G-CNN using a modified (relaxed) kernel consisting of a linear combination of standard G-CNN kernels 
{
Ψ
l
}
1
L
{Ψ 
l
​
 } 
1
L
​
 . Consider 
G
:
=
Z
n
⋊
H
G:=Z 
n
 ⋊H. Then, relaxed G-equivariant group convolution is defined as:

  
 
 
or equivalently as a linear combination of regular group convolutions with different kernels:

 
  
 
 
 
This second formulation makes for a more interpretable visualization, as one can see in the following figure. There, one can observe how a network might learn to downweight the feature maps corresponding to 180 degree rotations, thus breaking rotational equivariance and allowing for different processing of images picturing 6s and 9s.


Visualization of relaxed lifting convolutions (
L
=
1
L=1) as template matching. An input image 
f
in
f 
in
​
  contains a pattern 
e
e in different orientations, each of which is weighted differently by the model.
Relaxed steerable G-CNN
Relaxed steerable G-CNN modified steerable G-CNN in a similar manner. Again, let the kernel in convolution be a linear combination of other kernels, such that the weights used depend on the variable of integration, leading to loss of equivariance.

 
 
Furthermore, 
[3]
 introduces a regularization term to impose equivariance on both relaxed models mentioned above. In our experiments, however, the best-performing models were those without this term.

Methodology
Datasets
Super-Resolution
The data consists of liquid flowing in 3D space and is produced by a high-resolution state-of-the-art simulation hosted by the John Hopkins University 
[6]
 . Importantly, this dataset is forced to be isotropic, i.e. fully equivariant to rotations, by design.

For the experiment, a subset of 50 timesteps are taken, each downsampled from 
1
0
2
4
3
1024 
3
  to 
6
4
3
64 
3
  and processed into a task suitable for learning. The model is given an input of 3 consecutive timesteps, 
t
,
t
+
1
,
t
+
2
t,t+1,t+2 (which are first downsampled to 
1
6
3
16 
3
 ), and is tasked to upsample timestep 
t
+
1
t+1 to 
6
4
3
64 
3
 , see Figure 1 for a visualization.

We use the following 
3
3 models from 
[1]
’s experiment on the same dataset in Results:

CNN.
Regular G-CNN.
Relaxed G-CNN.
Figure 1
Figure 1: Super Resolution architecture, taken from [1].

Smoke Plume
This is a synthetic 
6
4
×
6
4
64×64 2D smoke simulation dataset generated by PhiFlow 
[7]
, where dispersion of smoke in a scene starting from an inflow position with a buoyant force is simulated (Figure 2).

The dataset we used has a fixed inflow with buoyant force only pointing in one of the following 
4
4 directions: upwards, downwards, left, or right. For our experiments we keep the buoyant force the same in all directions such that the data is fully equivariant w.r.t. 
9
0
90 degree rotations.

Figure 2
Figure 2: Example of a Smoke Plume sequence generated by PhiFlow.

The models trained on this dataset are tasked with predicting the upcoming frame based on the current one. We use the following 
2
2 models in Results:

Relaxed steerable G-CNN from 
[3]
 with relaxed equivariance w.r.t the C4 group.
Steerable G-CNN from 
[8]
 with full equivariance w.r.t the C4 group.
Training Dynamics Evaluation
To assess the training dynamics of a network, we are interested in the final performance and the generalizability of the learned parameters, which are quantified by the final RMSE, and the sharpness of the loss landscape near the final weight-point proposed in 
[9]
.

Sharpness
To measure the sharpness of the loss landscape after training, we consider changes in the loss averaged over random directions. Let 
D
D denote a set of vectors randomly drawn from the unit sphere, and 
T
T a set of displacements, i.e. real numbers. Then, the sharpness of the loss 
L
L at a point 
w
w is:

 
  
 
This definition is an adaptation from the one in 
[9]
. A sharper loss landscape around the model’s final weights usually implies a greater generalization gap.

Hessian Eigenvalue
Finally, the Hessian eigenvalue spectrum 
[2]
 sheds light on both the efficiency and efficacy of neural network training. Negative Hessian eigenvalues indicate a non-convex loss landscape, which can disturb the optimization process, whereas very large eigenvalues indicate training instability, sharp minima and consequently poor generalization.

Results
In this section, we study how equivariance imposed on a network influences the convexity of the loss landscape and generalization, answering all the subquestions posed in Introduction.

Smoke Plume with full Equivariance
First, we examine the training, validation and test RMSE for the Steerable G-CNN (E2CNN) 
[3]
 and Relaxed Steerable G-CNN (Rsteer) 
[8]
 models on the fully equivariant Smoke Plume dataset.

Figure 5
Figure 5: Train RMSE curve for rsteer and E2CNN models

Figure 6
Figure 6: Validation RMSE curve for rsteer and E2CNN models

Figure 7
Figure 7: Test RMSE for best models, averaged over five seeds

Figures 5 and 6 show the train and validation RMSE curves. While rsteer and E2CNN perform similarly on the training data, rsteer has lower RMSE on the validation data, indicating better generalization. Figure 7 confirms that rsteer performs best on the test set, consistent with results on the Isotropic Flow dataset in 
[1]
.

To understand why relaxed equivariant models outperform fully equivariant ones, we examine the sharpness of the loss and the Hessian spectra.

Figure 10
Figure 8: Sharpness at early and best epochs for rsteer and E2CNN models. On the equivariant Smokeplume dataset

Figure 10 shows that the rsteer model has much lower sharpness of the loss landscape compared to E2CNN for both checkpoints. This indicates a lower generalization gap, and thus more effective learning. This matches the lower validation RMSE curve we saw earlier.



A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
Figures 9 and 10 show Hessian spectra for the same checkpoints as the previous analysis. Regarding loss landscape flatness, both plots indicate that E2CNN has much larger eigenvalues than rsteer, potentially leading to training instability, less flat minima, and poor generalization for E2CNN.

To evaluate the convexity of the loss landscape, we examine the negative eigenvalues in the Hessian spectra. Neither model shows any negative eigenvalues, suggesting that both E2CNN and rsteer encounter convex loss landscapes. Therefore, convexity does not seem to significantly impact performance in this case.

Super Resolution
Similarly, we also analyze the training dynamics of the superresolution models on the isotropic Super-Resolution dataset.

First, we examine the training and validation MAE curves for the Relaxed Equivariant (RGCNN), Fully Equivariant (GCNN), and non-equivariant (CNN) models (run on 6 different seeds).

Figure 8
Figure 11: Training MAE curve for RGCNN, GCNN and CNN models

Figure 9
Figure 12: Validation MAE curve for RGCNN, GCNN and CNN models

Here, we observe that early in the training (around epoch 
3
3), RGCNN starts outperforming the other two models and keeps this lead until its saturation at around 
0
.
1
0.1 MAE. For this reason, we take a checkpoint for each model on epoch 
3
3 (early) and on its best epoch (Best), to examine the corresponding sharpness values.

Figure 10
Figure 13: Sharpness of the loss landscape on the super resolution dataset. Ran over 6 seeds, error bars represent the standard deviation. For early, the third epoch was chosen, while for best the epoch with the best validation loss was chosen.

Figure 13 shows that the relaxed model has the lowest sharpeness in both cases. This indicates that the relaxed steerable GCNN has better generalisability during its training and at its convergence, matching our findings on the previous dataset.

Concluding Remarks
We reproduced and extended the relevant findings in 
[1]
 reaffirming the effectiveness of relaxed equivariant models and demonstrating that they are able to outperform fully equivariant models even on perfectly equivariant datasets.

We furthermore investigated the authors’ speculation that this superior performance could be due to relaxed models having enhanced training dynamics. Our experiments empirically support this hypothesis, showing that relaxed models exhibit lower validation error, a flatter loss landscape around the final weights, and smaller Hessian eigenvalues, all of which are indicators of improved training dynamics and better generalization.

Our results suggest that replacing fully equivariant networks with relaxed equivariant networks could be advantageous in all application domains where some level of model equivariance is desired, including those where full equivariance is beneficial. For future research, we should investigate different versions of the relaxed model to find out which hyperparameters, like the number of filter banks, correlate with sharpness. Additionally, the method should be applied to different types of data to see if the same observations can be made there.

Code
Code and experiments for this blog
gconv, a PyTorch library for (relaxed) regular GCNNs
JHTDB 🤗 HuggingFace Dataset
References
Relaxed Octahedral Group Convolution for Learning Symmetry Breaking in 3D Physical Systems  [link]
Wang, R., Walters, R. and Smidt, T., 2023. NeurIPS 2023 AI for Science Workshop.
How Do Vision Transformers Work?  [PDF]
Park, N. and Kim, S., 2022.
Approximately Equivariant Networks for Imperfectly Symmetric Dynamics  [PDF]
Wang, R., Walters, R. and Yu, R., 2022.
The Cityscapes Dataset for Semantic Urban Scene Understanding  [PDF]
Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S. and Schiele, B., 2016.
Fast, Expressive SE
(
n
)
(n) Equivariant Networks through Weight-Sharing in Position-Orientation Space  [PDF]
Bekkers, E.J., Vadgama, S., Hesselink, R.D., Linden, P.A.v.d. and Romero, D.W., 2024.
A public turbulence database cluster and applications to study Lagrangian evolution of velocity increments in turbulence  [link]
Yi Li, E.P. and Eyink, G., 2008. Journal of Turbulence, Vol 9(), pp. N31. Taylor \& Francis. DOI: 10.1080/14685240802376389
Learning to Control PDEs with Differentiable Physics  [PDF]
Holl, P., Koltun, V. and Thuerey, N., 2020.
General 
E
(
2
)
E(2)-Equivariant Steerable CNNs  [PDF]
Weiler, M. and Cesa, G., 2021.
Improving Convergence and Generalization Using Parameter Symmetries  [PDF]
Zhao, B., Gower, R.M., Walters, R. and Yu, R., 2024.
© Copyright 2025 .

