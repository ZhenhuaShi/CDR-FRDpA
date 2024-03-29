# CDR-FRDpA
source code for CDR-FRDpA paper

CDR-FRDpA enhances the FCM-RDpA (Fuzzy C-Means Clustering, Regularization, DropRule, and Powerball AdaBelief; [paper](https://authors.elsevier.com/c/1dHuD4ZQE4EJG)|[arxiv](https://arxiv.org/abs/2012.00060)|[code](https://github.com/ZhenhuaShi/FCM-RDpA)|[blog](http://blog.sciencenet.cn/blog-3418535-1260629.html)) via consistent dimensionality reduction to optimize TSK fuzzy systems for regression in high dimensionality

run [demoCDR.m](https://github.com/ZhenhuaShi/CDR-FCM-RDpA/blob/main/demoCDR.m) to reproduce the results on the Estate-costs dataset of Fig.2/3 in the paper.

run [demoPS.m](https://github.com/ZhenhuaShi/CDR-FCM-RDpA/blob/main/demoPS.m) to reproduce the results on the Estate-costs dataset of Fig.4 in the paper.

run [demoInit.m](https://github.com/ZhenhuaShi/CDR-FCM-RDpA/blob/main/demoInit.m) to reproduce the results on the Estate-costs dataset of Fig.5 in the paper.

run [demoMF.m](https://github.com/ZhenhuaShi/CDR-FCM-RDpA/blob/main/demoMF.m) to reproduce the results on the Estate-costs dataset of Fig.7 in the paper.

run [demoRP.m](https://github.com/ZhenhuaShi/CDR-FCM-RDpA/blob/main/demoRP.m) to reproduce the results on the Estate-costs dataset of Fig.8 in the paper.

We also provide a [sugfis_mbgd_app.mlapp](https://github.com/ZhenhuaShi/CDR-FCM-RDpA/blob/main/sugfis_mbgd_app.mlapp) for simple test. Some examples are given as below:

## FCM-RDpA
<div align=center><img src="https://github.com/ZhenhuaShi/CDR-FRDpA/blob/main/FCM-RDpA.PNG"/></div>

## CDR-FRDpA
<div align=center><img src="https://github.com/ZhenhuaShi/CDR-FRDpA/blob/main/CDR-FRDpA.PNG"/></div>

## CDR-GRDpA
<div align=center><img src="https://github.com/ZhenhuaShi/CDR-FRDpA/blob/main/CDR-GRDpA.PNG"/></div>

## CDRP-FRDpA
<div align=center><img src="https://github.com/ZhenhuaShi/CDR-FRDpA/blob/main/CDRP-FRDpA.PNG"/></div>

## Citation
```
@Article{Shi2021,
  author  = {Zhenhua Shi and Dongrui Wu and Chenfeng Guo and Changming Zhao and Yuqi Cui and Fei-Yue Wang},
  journal = {Information Sciences},
  title   = {{FCM-RDpA}: {TSK} Fuzzy Regression Model Construction Using Fuzzy C-Means Clustering, Regularization, {D}rop{R}ule, and {P}owerball {A}da{B}elief},
  year    = {2021},
  pages   = {490-504},
  volume  = {574},
}
@Article{Wu2020,
  author  = {Dongrui Wu and Ye Yuan and Jian Huang and Yihua Tan},
  journal = {IEEE Trans. on Fuzzy Systems},
  title   = {Optimize {TSK} Fuzzy Systems for Regression Problems: Mini-batch Gradient Descent With Regularization, {D}rop{R}ule, and {A}da{B}ound ({MBGD-RDA})},
  year    = {2020},
  number  = {5},
  pages   = {1003-1015},
  volume  = {28},
}
```
