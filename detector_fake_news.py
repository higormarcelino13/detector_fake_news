# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('fake_and_real_news_dataset.csv')

df = df.sample(frac=1).reset_index(drop=True)

textos = df['text']
rotulos = df['label']

textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=0.3, random_state=42)

# Vetorização dos textos (tokenização e contagem de frequência)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_treino = vectorizer.fit_transform(textos_treino)
tfidf_teste = vectorizer.transform(textos_teste)

clf = MultinomialNB()
clf.fit(tfidf_treino, rotulos_treino)

predicoes = clf.predict(tfidf_teste)
acuracia = accuracy_score(rotulos_teste, predicoes)
print(f"Certeza: {acuracia:.3f}")

matriz_confusao = confusion_matrix(rotulos_teste, predicoes)

vp = matriz_confusao[1, 1]
fp = matriz_confusao[0, 1]
vn = matriz_confusao[0, 0]
fn = matriz_confusao[1, 0]

categorias = ['Verdadeiro Positivo (VP)', 'Falso Positivo (FP)', 'Verdadeiro Negativo (VN)', 'Falso Negativo (FN)']
contagens = [vp, fp, vn, fn]

plt.figure(figsize=(10,6))
plt.bar(categorias, contagens, color=['green', 'red', 'green', 'red'])
plt.title('Resultados do Modelo de Detecção de Fake News')
plt.xlabel('Categorias')
plt.ylabel('Contagem')
plt.show()
