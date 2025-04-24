# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("train_data.csv", index_col=['Id'], na_values="?")
df.head()


# %%
df.info()

# %% [markdown]
# 14 variáveis diferentes, treze independentes e uma de classe
# variável de classe é aquilo que queremos prever
# 6 dessas variáveis são numéricas, enquanto as 7 outras são categóricas
# as númericas são mais fácies de trabalhar e do computador compreender
# enquanto que as demais precisam ser transformadas em número

# %%
# estatísticas dos dados numéricos
df.describe()

# %% [markdown]
# Se o primeiro quartil, mediana e terceiro quartil são próximos (Q3-Q1/mediana < 0.5), isso indica uma distribuição de dados concentrada, pouco espalhada
# Aqui podemos ver que capital gain e loss estão bem concentradas, pois os 3 valores são iguais
# 
# Se a média for muito diferente da mediana (mean e 50%), pode indicar uma distribuição assimétrica (abs(média-mediana)/mediana > 0.1 indica assimetria relevante, acima de 0.3 é uma assimetria forte)
# Novamente comportamento percebido no gain e loss
# 
# Se std muito alto (std/media > 0.5 alta, > 1.0 muito alta), indica que os dados estão bem dispersos, novamente gain e loss se destacam
# 
# Gain parece ter outliers devido ao seu grande desvio padrão e seu máximo de 99999

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# Vamos transformar a colune income para valores binários 0 e 1 para podermos trabalhar mais facilmente com a variável de classe

# %%
df_analysis = df.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_analysis['income'] = le.fit_transform(df_analysis['income'])
df_analysis['income']

# %%
df_numeric = df_analysis.select_dtypes(include=[np.number])
df_numeric.info()

# %%
mask = np.triu(np.ones_like(df_numeric.corr(), dtype=np.bool_))

plt.figure(figsize=(10,10))

sns.heatmap(df_numeric.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1, cmap='autumn')
plt.show()

# %% [markdown]
# Podemos ver que tdas as variáveis possuem algum grau de correlação com "income", menos "fnlwgt" com correlação -0.0095
# Parece não ter valor em usar essa variável para nosso modelo

# %%
sns.catplot(x='income', y='age', kind="boxen", data=df_analysis)

# %%
sns.catplot(x='income', y='hours.per.week', kind="boxen", data=df_analysis)

# %%
sns.catplot(x='income', y='fnlwgt', kind="boxen", data=df_analysis)

# %%
sns.catplot(x='income', y='education.num', kind="boxen", data=df_analysis)

# %%
sns.catplot(x='income', y='capital.gain', kind="boxen", data=df_analysis)

# %%
sns.catplot(x='income', y='capital.gain', data=df_analysis, kind='strip')

# %%
sns.catplot(x='income', y='capital.loss', kind="boxen", data=df_analysis)

# %%
sns.catplot(x='income', y='capital.loss', data=df_analysis)

# %% [markdown]
# Com esses gráficos podemos ter uma certeza maior de que o peso realmente não importa
# 
# Além disso, capital.gain e .loss de fato estão bem esparsas e possuem outliers. Essa informação é importante para nossa análise, mas os dados precisam ser pré-processados

# %%
sns.catplot(y='sex', x='income', kind='bar', data=df_analysis)

# %% [markdown]
# Esse gráfico indica que 10% das mulheres ganham mais de 50k e 30% dos homens ganham mais

# %%
sns.catplot(y='marital.status', x='income', kind='bar', data=df_analysis)

# %%
sns.catplot(y='occupation', x='income', kind='bar', data=df_analysis)

# %%
sns.catplot(y='relationship', x='income', kind='bar', data=df_analysis)

# %%
sns.catplot(y='race', x='income', kind='bar', data=df_analysis)

# %%
sns.catplot(y='native.country', x='income', kind='bar', data=df_analysis)

# %% [markdown]
# Todas as variáveis parecem estar ligadas com a renda das pessoas
# 
# As linhas pretas representam a variação de renda dentro de cada grupo. A maioria ds países possui uma variação muito grande, tão grande que não podemos ter clareza de seu valore real

# %%
df_analysis['native.country'].value_counts()

# %% [markdown]
# Podemos lidar com esse problema (ver se algum outro tratou isso ou só retirou os dados também)

# %% [markdown]
# Limpeza dos dados

# %%
df = pd.read_csv("train_data.csv", index_col=['Id'], na_values="?")
df.head()

# %%
df.drop_duplicates(keep='first', inplace=True)

# %%
df = df.drop(['fnlwgt', 'native.country'], axis=1)

# %%
df.head()

# %%
Y_train = df.pop('income')

X_train = df

# %%
X_train.head()

# %% [markdown]
# Pré-processamento dos dados

# %%
numerical_cols = list(X_train.select_dtypes(include=[np.number]).columns.values)

numerical_cols.remove('capital.gain')
numerical_cols.remove('capital.loss')

sparse_cols = ['capital.gain', 'capital.loss']

categorical_cols = list(X_train.select_dtypes(exclude=[np.number]).columns.values)

print("Colunas numéricas: ", numerical_cols)
print("Colunas esparsas: ", sparse_cols)
print("Colunas categóricas: ", categorical_cols)

# %% [markdown]
# Dados categóricos

# %%
from sklearn.impute import SimpleImputer

simple_imputer = SimpleImputer(strategy='most_frequent')
# testar com mediana e média também

# %%
from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(drop='if_binary')
# transforma os n categorias em n colunas, sendo 1 na coluna correspondente
# a informação e 0 nas demais

# %%
from sklearn.pipeline import Pipeline

categorial_pipeline = Pipeline(steps= [
    ('imputer', simple_imputer),
    ('onehot', one_hot)
])

# %% [markdown]
# Dados numéricos

# %%
from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=10, weights="uniform")
# usa o dado faltante como variável de classe e usa knn para decidir
# o dado faltante

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# média 0 e desvio padrão 1 para todas as variáveis

# %%
numerical_pipeline = Pipeline(steps= [
    ('imputer', knn_imputer),
    ('scaler', scaler)
])

# %% [markdown]
# Dados esparsos

# %%
from sklearn.preprocessing import RobustScaler

sparse_pipeline = Pipeline(steps=[
    ('imputer', knn_imputer),
    ('scaler', RobustScaler())
])

# %%
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('spr', sparse_pipeline, sparse_cols),
    ('cat', categorial_pipeline, categorical_cols),
])

# %%
X_train = preprocessor.fit_transform(X_train)

# %% [markdown]
# Predição

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=20)

# %% [markdown]
# Cross-validation com os dados

# %%
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score


scorers = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label='>50K'),
    "recall": make_scorer(recall_score, pos_label='>50K')
}

for metric_name, scorer in scorers.items():
    scores = cross_val_score(knn, X_train, Y_train, cv=50, scoring=scorer)
    print(f"{metric_name.capitalize()} com cross-validation: {scores.mean():.4f}")

# %% [markdown]
#   Acc | Pre | Recall
# 0.8683 & 0.7712 & 0.6449 & sem as colunas ['fnlwgt', 'native.country', 'education], most_frequent e 10nn e 20nn final, cv 5 \\
# 0.8644 & 0.7586 & 0.6414 & com todas as colunas, most_frequent e 10nn e 20nn final \\
# 0.8645 & 0.7588 & 0.6415 & com todas as colunas sem apagar dados duplicados \\
# 0.8374 & 0.7713 & 0.4619 & sem pipeline de dados categóricos  \\
# 0.8335 & 0.6833 & 0.5752 & sem pipeline de dados esparsos  \\
# 0.8573 & 0.7586 & 0.5980 & sem pipeline numérica  \\
# 0.8256 & 0.6828 & 0.5151 & sem pipeline numérica e de dados  esparsos \\
# 0.8291 & 0.9523 & 0.3054 & sem pipeline numérica e de dados  categóricos \\
# 0.7910 & 0.6161 & 0.3508 & sem pipeline de dados esparsos e  categóricos \\
# 0.8682 & 0.7692 & 0.6478 & filtros e limpagens, cv 10  \\
# 0.6459 & 0.7680 & 0.6459 & filtros e limpagens, cv 50  \\

# %% [markdown]
# Usando os dados de teste

# %%
knn.fit(X_train, Y_train)

# %%
test_data = pd.read_csv("test_data.csv", index_col=['Id'], na_values="?")

# %%
X_test = test_data.drop(['fnlwgt', 'native.country', 'education'], axis=1)

# %%
X_test = preprocessor.transform(X_test)

# %%
predictions = knn.predict(X_test)


