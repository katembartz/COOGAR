# COOGAR
A Continuous Optimization Approach for Graph Cuts-based Phase Unwrapping

If you use this code in your work, please cite (currently in press):

Publication:
```
Bartz, K.M., Wei, S., Remedios, S.W., Zhang, J., Carass, A., Dewey, B.E., Prince, J.L., Trzasko, J.D. (_in press_, 2027). A Continuous Optimization Approach for Graph Cuts-based Phase Unwrapping. In: Graphs in Biomedical Image Analysis. GRAIL 2026. Lecture Notes in Computer Science. Springer, Cham.
```

Citation:
```
@inproceedings{bartz2027_COOGAR_inpress,
  title = {A Continuous Optimization Approach for Graph Cuts-based Phase Unwrapping},
  author = {Bartz, Kathleen M. and Wei, Shuwen and Remedios, Samuel W. and Zhang, Jinwei and Carass, Aaron and Dewey, Blake E. and Prince, Jerry L. and Trzasko, Joshua D.),
  booktitle = {Graphs in Biomedical Image Analysis},
  year = {2027},
  publisher = {Springer Nature Switzerland},
  address = {Cham}
}
```
The citation information will be updated once the proceedings for GRAIL MICCAI are finalized.

# Quick Start

We recommend starting from a fresh python installation.

```
conda create -n coogar python=3.10
conda activate coogar
```

Clone and install this repository.

```
git clone https://github.com/katembartz/COOGAR
cd COOGAR
pip install .
```

Run COOGAR on your phase volume.

```
coogar \
  --input {path_to_nibabel_volume} \
  [--out-dir {path_to_store_results}] \
  [--tmp-dir {path_to_store_intermediate_results}] \
  [--mask {path_to_ROI_mask}] \
  [--outerIter {int}] \
  [--p {int}] \
  [--gpu-id {int}] \
  [--padding {int}]
```

| Parameter | Data Type | Usage |
| :---     | :---    | :---     |
| --input | Path to nibabel image | image to unwrap |
| --out-dir | Path to directory | (optional) directory to place unwrapped prediction |
| --tmp-dir | Path to directory | (optional) directory to store intermediate results |
| --mask | Path to nibabel image | (optional) mask for ROI to unwrap |
| --outerIter | int | (optional) number of graph cuts to apply |
| --p | int | (optional) Lp regularization |
| --gpu-id | int | (optional) specify for GPU usage |
| --padding | int | (optional) padding to reduce border artifacts |




# Updates

We are actively working to make this software more general and user-friendly.  All updates to this repository will be posted and time stamped here. 

+ 09.23.2026: COOGAR version 1.0.0: 3D Phase Unwrapping Method

Note that v1.0.0 is currently hard coded for 3D Phase Images. A future version will be coded for arbitrary nD volumes, and an update will be posted here once the version is available.
