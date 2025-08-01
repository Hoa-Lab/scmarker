import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy.stats import pearsonr
import tensorflow.keras as keras

#-------------------------------------------------
class seq_dense:
    def __init__(self, n_layer=1, r=0.1, lr=0.0001, activation='swish', l1=0, l2=0):
        self.layer = n_layer
        #self.node=n_gene*2
        self.r = r
        self.lr = lr
        self.acti = activation
        self.l1 = l1
        self.l2 = l2
        return

    def build(self, n_in, n_out):
        #model
        reg = keras.regularizers.L1L2(l1=self.l1, l2=self.l2)
        model = keras.models.Sequential()
        model.add(keras.layers.Input((n_in,)))
        for i in range(self.layer):
            model.add(keras.layers.Dropout(self.r))
            model.add(keras.layers.Dense(n_in*2, activation=self.acti, kernel_initializer='he_normal', kernel_regularizer=reg))
        model.add(keras.layers.Dense(n_out, activation='softmax'))
        print(model.summary())
        #compile
        opt = keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
        return model

def get_pred(ada,
    label: str = 'cell',   # column name of cell type information
    l_gene: list = [],     # customized gene list for model training. if used, "n_gene" will be ignored 
    n_gene: int = 2000,    # number of genes used for model training
    n_epoch: int = 5,
    n_layer: int = 1,
    dropout: float = 0.1,
    lr: float = 0.0001,    # learning rate
    activation: str = 'swish',
    batch: int = 32,
    vs: float = 0.1        # validation split ratio
    ) -> pd.DataFrame:
    #cell
    l_cell = ada.obs[label].astype('str').unique().tolist()
    if len(l_cell) < 2:
        print('not enough cell types...')
        return
    #gene
    if len(l_gene) == 0:
        df = ada.to_df().T
        df['r'] = (df>0).mean(axis=1)
        df = df.sort_values('r', ascending=False)
        l_gene = df.index.tolist()[:n_gene]
    #X
    df = ada.to_df().reindex(l_gene, axis=1).dropna(axis=1)
    if df.shape[1] == 0: 
        print('not enough genes...')
        return
    X = df.values
    n_in = df.shape[1]
    #y
    df['y'] = pd.Categorical(ada.obs[label], categories=l_cell, ordered=True)
    y = df['y'].cat.codes
    y = keras.utils.to_categorical(y)
    df = df.drop('y', axis=1)
    #model
    n_out = len(l_cell)
    model = seq_dense(n_layer=n_layer, r=dropout, lr=lr, activation=activation)
    model = model.build(n_in, n_out)
    model.fit(X, y, epochs=n_epoch, batch_size=batch, validation_split=vs, callbacks=[])
    #pred
    df_pred = pd.DataFrame(model.predict(X), columns=l_cell, index=df.index)
    return df_pred

def get_score(ada, 
    df_pred: pd.DataFrame,
    cell: str
    ) -> pd.DataFrame:
    df = ada.to_df()
    df_corr = pd.DataFrame(df.corrwith(df_pred[cell], axis=0), columns=[f'{cell}_score'])
    df_corr = df_corr.sort_values(f'{cell}_score', ascending=False)
    return df_corr


