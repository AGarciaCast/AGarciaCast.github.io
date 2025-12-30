---
title: 'Symmetry-Aware Graph Metanetwork Autoencoders: Model Merging through Parameter
  Canonicalization'
authors:
- Odysseas Boufalis
- Jorge Carrasco-Pollo
- Joshua Rosenthal
- Eduardo Terres-Caballero
- Alejandro Garcı́a-Castellanos
date: '2025-01-01'
publishDate: '2025-12-30T18:05:25.184976Z'
publication_types:
- article-journal
publication: '*arXiv preprint arXiv:2511.12601*'
abstract: Neural network parameterizations exhibit inherent symmetries that yield
  multiple equivalent minima within the loss landscape. Scale Graph Metanetworks (ScaleGMNs)
  explicitly leverage these symmetries by proposing an architecture equivariant to
  both permutation and parameter scaling transformations. Previous work by Ainsworth
  et al. (2023) addressed permutation symmetries through a computationally intensive
  combinatorial assignment problem, demonstrating that leveraging permutation symmetries
  alone can map networks into a shared loss basin. In this work, we extend their approach
  by also incorporating scaling symmetries, presenting an autoencoder framework utilizing
  ScaleGMNs as invariant encoders. Experimental results demonstrate that our method
  aligns Implicit Neural Representations (INRs) and Convolutional Neural Networks
  (CNNs) under both permutation and scaling symmetries without explicitly solving
  the assignment problem. This approach ensures that similar networks naturally converge
  within the same basin, facilitating model merging, i.e., smooth linear interpolation
  while avoiding regions of high loss. The code is publicly available on our GitHub
  repository.
links:
- name: URL
  url: https://arxiv.org/pdf/2511.12601
---
