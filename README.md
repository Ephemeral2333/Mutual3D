# Mutual3D

Official repository for **Mutual3D: Generative Instance Mesh Reconstruction via Semantic–Geometric Mutual Refinement**.

Mutual3D is an end-to-end framework for scene-level instance mesh reconstruction from raw point clouds. It jointly performs instance perception and object-level mesh reconstruction through bidirectional semantic–geometric refinement.

Semantic predictions guide generative shape reconstruction, while reconstructed geometry provides structural feedback to refine instance understanding. Mutual3D further adopts a diffusion-based implicit reconstruction module to recover complete and plausible object meshes from partial and occluded observations.

## News

Complete code is coming soon.

## Method

Mutual3D consists of three main components:

* Query-based instance perception for predicting instance masks, categories, and 3D boxes.
* Diffusion-based implicit reconstruction for generating complete object shapes from partial observations.
* Geometry-aware query refinement for feeding reconstructed shape priors back into instance perception.

## Results

Mutual3D achieves strong performance on indoor scene instance mesh reconstruction benchmarks and produces more complete, coherent, and well-bounded instance meshes, especially in cluttered and occluded scenes.

More results, pretrained models, and evaluation scripts will be released soon.