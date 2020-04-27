# Perturbed-Masking

This repo contains code to replicate experiments in the ACL'2020 Paper

**Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT**

Note that this not the original code we used when working on the paper. Since the huggingface transformers repo has been updated a lot (0.5.0 -> 2.5.1)

We cleaned and migrated the codes so that it works under the latest version of huggingface transformers and other packages that we used.

If you encountered with any problem, please report in "Issues".


## Download dataset

Since some of the dataset is not publicly available (E.g., PTB), we cannot include them in the repo. Please follow the instructions below to prepare the dataset:

##### Dependency

See papar for preprocess details.

##### Constituency
Down load the Penn Treebank corpus from LDC, and follow the dummy dirs/files in 'constituency/data' to organize the files.

##### Discourse

Download the SciDTB dataset from https://github.com/PKU-TANGENT/SciDTB, put the folder under 'discourse/'

## Dependency Probe
- Get impact matrix, take PUD dataset for example
```
python preprocess.py --cuda \
    --probe dependency \
    --data_split PUD \
    --dataset ./dependency/data/PUD.conllu 
```
- Now we do parsing, using Eisner algo
```
python parsing.py --probe dependency \
    --matrix ./results/dependency/bert-dist-PUD-12.pkl \
    --decoder eisner
```  

Run the above command should give you an UAS of 41.7 (See Paper Tabel 1)

## Constituency Probe
- Get impact matrix, take WSJ23 dataset for example
```
python preprocess.py --cuda \
    --probe constituency \
    --data_split WSJ23 \
    --dataset constituency/data/WSJ/ 
```
- Now we do parsing, using MART algo
```
python parsing.py --probe constituency \
    --matrix [the generated impact matrix file] \
    --decoder mart
```


## Discourse Probe

- Get impact matrix, take SciDTB test set for example
```
python preprocess.py --cuda \
    --probe discourse \
    --data_split SciDTB \
    --dataset ./discourse/SciDTB/test/gold/
```

- Now we do parsing, using Eisner algo
```
python parsing.py --probe discourse \
    --matrix [the generated impact matrix file] \
    --decoder eisner
```  


## BERT trees in downstream task (ABSC)

 To reproduce results in Tabel 5.
- Git clone the PWCN repo https://github.com/GeneZC/PWCN
- Follow their instructions to setup
- Replace files in 'PWCN/datasets/semeval14/' with those of same name from 'Perturbed-Masking/ABSC/'. We provide the distance file we generated for quite reproduction.  
- Follow their instruction to train the model.

In the training, PWCN use random initialization and report the averaged performance, thus when re-run our experiments on a different machine, we observe slightly different results. We report all results below:


|-| Model  | Laptop-Acc  | Laptop-F1  | Restaurant-Acc  | Restaurant-F1  |
|:---:|:---:|:---:|:---:|:---:|:---:|
| PWCN paper|PWCN+Dep |76.12|72.12|80.96|72.21|
|Machine 1|PWCN+Dep   |76.08|72.02|80.98|72.28|
|Machine 1|PWCN+Eisner|75.99|72.01|81.21|73.00|
|Machine 2|PWCN+Dep   |75.17|70.66|80.67|71.92|
|Machine 2|PWCN+Eisner|74.82|70.90|80.66|72.16|

