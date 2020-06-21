#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[79]:


import pandas as pd
import numpy as np


# In[80]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[81]:


##Verificando head do dataset
black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[82]:


def q1():
    return black_friday.shape
    # Retorne aqui o resultado da questão 1.


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[83]:


def q2():
    return (black_friday[black_friday['Gender'] == 'F']['Age'].value_counts()['26-35'].item())
    # Retorne aqui o resultado da questão 2.


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[84]:


def q3():
    return black_friday['User_ID'].nunique()
    # Retorne aqui o resultado da questão 3.


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[85]:


def q4():
    return black_friday.dtypes.nunique()
    # Retorne aqui o resultado da questão 4.


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[86]:


def q5():
    return float(black_friday[black_friday.isna().any(axis=1)].shape[0]/black_friday.shape[0])
    # Retorne aqui o resultado da questão 5.


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[87]:


def q6():
    return int(black_friday.isnull().sum().max())
    # Retorne aqui o resultado da questão 6.


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[88]:


def q7():
    return float(black_friday['Product_Category_3'][black_friday.Product_Category_3.notnull()].mode())
    # Retorne aqui o resultado da questão 7.


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[89]:


def q8():
    # Retorne aqui o resultado da questão 8.
    v = black_friday['Purchase']
    v_min = black_friday['Purchase'].min()
    v_max = black_friday['Purchase'].max()
    normal = (v - v_min) / (v_max - v_min)
    return float(normal.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[90]:


def q9():
    # Retorne aqui o resultado da questão 9.
    mean = black_friday['Purchase'].mean()
    std = black_friday['Purchase'].std()
    standardized = (black_friday['Purchase'] - mean)/(std)
    return int(((standardized >= -1) & (standardized <= 1)).sum())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[91]:


def q10():
    # Retorne aqui o resultado da questão 10.
    prod_cat_2 = black_friday['Product_Category_2'].isna()
    prod_cat_3 = black_friday['Product_Category_3'].isna()
    return bool(prod_cat_2[prod_cat_2 == True].equals(prod_cat_3[prod_cat_2 == True]))

