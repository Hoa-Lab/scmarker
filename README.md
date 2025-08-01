# Marker Gene Identification for scRNASeq Data 

## Introduction
 In this method, we defined the marker gene score of a given cell type as the Pearson correlation coefficient between the gene expression and the predicted cell type probability. The model used here to predict cell type probability is a dense model with the "*softmax*" function.

## Dependency
- Scipy
- Tensorflow

## Quick Start
- Get predicted cell type probability table:  
`import scmarker as scm`  
`df_pred = scm.get_pred(ada, 'cell')`  

  Note: "*get_pred*" function requires two arguments. "*ada*" is the input dataset in AnnData format. "*cell*" is the column name of the annotation information in the input data's "*obs*" attribute. 
  
 - calculate marker gene scores:  
`df = scm.get_score(ada, df_pred, 'cell type')`  

    Note: "*get_score*" function require three arguments. "*ada*" is the input dataset in AnnData format. "*df_pred*" is the predicted cell type probability table. "*cell type*" is the name of the interested cell type. "*get_score*" function returns a Pandas dataframe.

