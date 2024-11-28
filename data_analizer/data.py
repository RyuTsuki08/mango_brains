
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os  

print(os.listdir('.'))

data = pd.read_csv('./data_analizer/emotions.csv')
df = pd.DataFrame(data)
df.head()

#Distrubucion de los labels
X = df.drop(columns=['label'])
y = df['label']


# Normalizar los datos

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Convertir a DataFrame y añadir la etiqueta
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['label'] = y

# Visualizar
# plt.figure(figsize=(10, 6))
# for label in df_pca['label'].unique():
#     subset = df_pca[df_pca['label'] == label]
#     plt.scatter(subset['PC1'], subset['PC2'], label=label)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCA de EEG Brainwave Dataset')
# plt.legend()
# plt.show()



# ##Informacion Extra


#Crear datasets individuales para cada carateristica del general
columns = df.columns.tolist()

keywords = ['mean_d', 'mean', 'stddev', 'moments', 'max', 'min', 'covmat', 'eigen', 'logm', 'entropy', 'correlate', 'fft']

filtered_dfs = {}

for keyword in keywords:
    filtered_dfs[keyword] = df.loc[:, [col for col in columns if keyword in col]]

mean_d_col = [col for col in columns if 'mean_d' in col]
filtered_dfs['mean'] = filtered_dfs['mean'].drop(mean_d_col, axis=1)

mean_df = filtered_dfs['mean']
mean_d_df = filtered_dfs['mean_d']
stddev_df = filtered_dfs['stddev']
moments_df = filtered_dfs['moments']
max_df = filtered_dfs['max']
min_df = filtered_dfs['min']
covmat_df = filtered_dfs['covmat']
eigen_df = filtered_dfs['eigen']
logm_df = filtered_dfs['logm']
entropy_df = filtered_dfs['entropy']
correlate_df = filtered_dfs['correlate']
fft_df = filtered_dfs['fft']



# ##Al General_df se le peude sumar cualquera de los anteriores dependiendo de las necesidades 


#Dataset general concatenando los anteriorres necesarios 
General_df = pd.concat([max_df, min_df, fft_df], axis=1)
General_df = General_df.merge(df['label'], left_index=True, right_index=True)
Columns =  np.array(General_df.columns)



entropy = entropy_df.mean(axis=1)



General_df['entropy'] = entropy

#Control de columnas
# for col in General_df.columns:
#     print(col)


# ##Ejemplo masculino (ejemplo de uso)


#Dataset con ejemplos de datos del primer sujeto
Needed_columns = ['max_0_a', 'max_1_a', 'max_2_a', 'max_3_a', 'min_0_a', 'min_1_a', 'min_2_a', 'min_3_a', 'fft_0_a', 'fft_1_a', 'fft_2_a', 'fft_3_a', 'entropy']
Sample_df_0 = General_df[Needed_columns]



# figsize = (10, 8)
# plt.figure(figsize=figsize)
# sns.heatmap(Sample_df_0.corr(), annot=True)



Sample_df_0['label'] = df['label']
Sample_df_0.reset_index(drop=True)



# Crear el gráfico de barras
# plt.figure(figsize=(12, 6))
# sns.barplot(x='label', y='entropy', data=Sample_df_0, ci=None)
# plt.title('Media de la Entropía según los Labels')
# plt.xlabel('Label')
# plt.ylabel('Media de la Entropía')
# plt.show()


# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='max_0_a', y='entropy', hue='label', data=Sample_df_0, alpha = 0.5) # Changed df to Sample_df_0
# plt.title('Entropía vs. Max_0_a')
# plt.xlabel('Max_0_a')
# plt.ylabel('Entropía')
# plt.show()


# ###Ejemplo Femenino (Ejemplo de uso)


Needed_columns2 = ['max_0_b', 'max_1_b', 'max_2_b', 'max_3_b', 'min_0_b', 'min_1_b', 'min_2_b', 'min_3_b', 'fft_0_b', 'fft_1_b', 'fft_2_b', 'fft_3_b', 'entropy']
Sample_df_1 = General_df[Needed_columns2]
Sample_df_1['label'] = df['label']



# figsize = (10, 8)
# plt.figure(figsize=figsize)
# sns.heatmap(Sample_df_1.corr(), annot=True)


# Crear el gráfico de barras
# plt.figure(figsize=(12, 6))
# sns.barplot(x='label', y='entropy', data=Sample_df_1, ci=None)
# plt.title('Media de la Entropía según los Labels')
# plt.xlabel('Label')
# plt.ylabel('Media de la Entropía')
# plt.show()



# Grafico de violin para visualizar la distribución de las etiquetas
# Sample_df_0['index'] = Sample_df_0.index
# plt.figure(figsize=(12, 6))
# sns.violinplot(x='label', y='index', data=Sample_df_0, scale='width', inner='quartile')
# plt.title('Distribución de Labels a lo Largo del Dataset')
# plt.xlabel('Label')
# plt.ylabel('Índice')
# plt.gca().invert_yaxis()  #invertir el eje y para mostrar el inicio del dataset en la parte superior
# plt.show()



# #Analisis general


# sns.histplot(Sample_df_1["label"])


# Seleccionar algunas características de Fourier para la visualización
columnas_fft = ['fft_741_b', 'fft_742_b', 'fft_743_b', 'fft_744_b', 'fft_745_b', 'fft_746_b', 'fft_747_b', 'fft_748_b', 'fft_749_b']

# Filtrar el dataset por cada tipo de emoción
df_negative = df[df['label'] == 'NEGATIVE']
df_neutral = df[df['label'] == 'NEUTRAL']
df_positive = df[df['label'] == 'POSITIVE']



# Función para plotear las señales
def plot_signals(df, label, ax):
    for i in range(min(5, len(df))):  # Mostrar hasta 5 ejemplos por emoción
        signal = df.iloc[i][columnas_fft].values
        ax.plot(signal, label=f'Muestra {i+1}')
    ax.set_title(f'Señales de EEG - {label}')
    ax.set_xlabel('Componente de Frecuencia')
    ax.set_ylabel('Amplitud')
    ax.legend()

# Crear una figura y ejes con tamaño personalizado
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))  # Cambiar el tamaño aquí

# Suponiendo que df_negative, df_neutral y df_positive ya están definidos
# Plotear las señales para cada emoción
# plot_signals(df_negative, 'NEGATIVE', axes[0])
# plot_signals(df_neutral, 'NEUTRAL', axes[1])
# plot_signals(df_positive, 'POSITIVE', axes[2])

# plt.tight_layout()
# plt.show()



Sample_df = df.loc[713, 'fft_0_b':'fft_749_b']

# fig = plt.figure(figsize=(18, 9))
# plt.plot(Sample_df)
# plt.title('Sample EEG Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()


# Sample_df


mean_sample = mean_df.loc[0, 'mean_1_a':'mean_4_b']

# fig = plt.figure(figsize=(18, 9))
# plt.plot(mean_sample)
# plt.title('Sample EEG Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()


# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

# # Primer subplot
# axes[0].plot(df.loc[713, 'fft_0_b':'fft_749_b'[:50]], color = "orange")
# axes[0].set_title('Negative FT')
# axes[0].legend()

# # Segundo subplot
# axes[1].plot(df.loc[1463, 'fft_0_b':'fft_749_b'[:50]], color = "blue")
# axes[1].set_title('Neutral FT')
# axes[1].legend()

# # Tercer subplot
# axes[2].plot(df.loc[940, 'fft_0_b':'fft_749_b'[:50]], color = "green")
# axes[2].set_title('Positive FT')
# axes[2].legend()

# Ajustar el layout para evitar superposiciones
# plt.tight_layout()
# plt.show()


# fig = plt.figure(figsize=(18, 9))
# plt.plot(df.loc[713, 'fft_0_b':'fft_749_b'[:50]], label = 'Negative', alpha = 0.5)
# plt.plot(df.loc[1463, 'fft_0_b':'fft_749_b'[:50]], label = 'Neutral', alpha = 0.5)
# plt.plot(df.loc[940, 'fft_0_b':'fft_749_b'[:50]], label = 'Positive', alpha = 0.5)
# plt.title('EEG Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()


