import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Baixar stopwords
nltk.download('stopwords')

# Configurações de layout e tema
st.set_page_config(layout="wide")

# Função para carregar modelos e tokenizer com cache para otimização
@st.cache_resource
def load_models():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    kmeans = load('kmeans_model.joblib')
    return tokenizer, model, kmeans

tokenizer, model, kmeans = load_models()

# Carregar o dataset com valores originais de rating e year com cache para dados
@st.cache_data
def load_data():
    df = pd.read_csv('all_movies_with_clusters.csv')  # Dataset com os clusters
    bert_embeddings = np.load('bert_embeddings.npy')  # Embeddings das sinopses
    return df, bert_embeddings

df, bert_embeddings = load_data()

# Pré-processamento de texto (remoção de stopwords e stemming)
def preprocess_text(text):
    stop_words = set(stopwords.words('portuguese'))
    stemmer = SnowballStemmer('portuguese')
    
    # Tokenizar e processar
    tokens = text.lower().split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    
    # Unir novamente em uma string processada
    return ' '.join(tokens)

# Função para gerar embeddings de sinopse com cache
@st.cache_data
def get_bert_embedding(text, _tokenizer):
    text = preprocess_text(text)
    
    inputs = _tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.detach().numpy()

# Função para recomendar filmes com base no cluster
def recommend_movies(cluster_number, df, num_recommendations=5):
    cluster_movies = df[df['cluster'] == cluster_number]
    recommended_movies = cluster_movies.sort_values(by='rating_original', ascending=False).head(num_recommendations)
    return recommended_movies[['title_pt', 'rating_original', 'year_original']]

# Função para recomendar filmes com base no gênero
def recommend_by_genre(genres, df, num_recommendations=5):
    genre_movies = df[df['genre'].isin(genres)]
    recommended_movies = genre_movies.sort_values(by='rating_original', ascending=False).head(num_recommendations)
    return recommended_movies[['title_pt', 'rating_original', 'year_original']]

# Recomendação com base no distanciamento de vetor das sinopses
def recommend_movies_by_vector_distance(embedding, df, embeddings, num_recommendations=5):
    similarities = cosine_similarity(embedding, embeddings).flatten()
    similar_indices = np.argsort(similarities)[-num_recommendations:][::-1]
    recommended_movies = df.iloc[similar_indices]
    return recommended_movies[['title_pt', 'rating_original', 'year_original']]

# Função para lidar com os diferentes métodos de recomendação
def handle_recommendation(method):
    if method == "Método 1: Escolha de Sinopse":
        st.write("Escolha uma sinopse abaixo que mais lhe agrada:")
        random_movies = df.sample(5)
        sinopse_choices = random_movies['sinopse'].tolist()
        sinopse_selecionada = st.radio("Escolha uma sinopse:", sinopse_choices)
        
        if st.button("Recomendar Filmes"):
            with st.spinner('Carregando recomendações...'):
                selected_movie = random_movies[random_movies['sinopse'] == sinopse_selecionada]
                selected_cluster = selected_movie['cluster'].values[0]
                recommendations = recommend_movies(selected_cluster, df)
            st.write(f"Filmes recomendados para você (Cluster {selected_cluster}):")
            for index, row in recommendations.iterrows():
                st.write(f"{row['title_pt']} - Rating: {row['rating_original']} - Ano: {row['year_original']}")
    
    elif method == "Método 2: Escreva sua Própria Sinopse (Cluster)":
        st.write("Escreva uma sinopse de um filme que você gostaria:")
        user_input = st.text_area("Digite sua sinopse aqui")
        
        if st.button("Recomendar Filmes"):
            with st.spinner('Gerando embedding da sua sinopse...'):
                user_embedding = get_bert_embedding(user_input, tokenizer)
                user_cluster = kmeans.predict(user_embedding)
                recommendations = recommend_movies(user_cluster[0], df)
            st.write(f"Filmes recomendados para você (Cluster {user_cluster[0]}):")
            cols = st.columns(3)
            for index, row in recommendations.iterrows():
                with cols[index % 3]:
                    st.write(f"**{row['title_pt']}**")
                    st.write(f"Rating: {row['rating_original']}")
                    st.write(f"Ano: {row['year_original']}")
    
    elif method == "Método 3: Escreva sua Própria Sinopse (Proximidade)":
        st.write("Escreva uma sinopse de um filme que você gostaria:")
        user_input = st.text_area("Digite sua sinopse aqui")
        
        if st.button("Recomendar Filmes"):
            with st.spinner('Calculando a similaridade com outros filmes...'):
                user_embedding = get_bert_embedding(user_input, tokenizer)                
                recommendations = recommend_movies_by_vector_distance(user_embedding, df, bert_embeddings)
                
            st.write("Filmes recomendados com base na proximidade das sinopses:")
            cols = st.columns(3)
            for index, row in recommendations.iterrows():
                with cols[index % 3]:
                    st.write(f"**{row['title_pt']}**")
                    st.write(f"Rating: {row['rating_original']}")
                    st.write(f"Ano: {row['year_original']}")

    elif method == "Método 4: Escolha por Gênero":
        st.write("Selecione um ou mais gêneros de filmes:")
        unique_genres = df['genre'].unique().tolist()
        selected_genres = [genre for genre in unique_genres if st.checkbox(genre)]
        
        if st.button("Recomendar Filmes"):
            if selected_genres:
                with st.spinner('Buscando filmes recomendados...'):
                    recommendations = recommend_by_genre(selected_genres, df)
                st.write(f"Filmes recomendados para você nos gêneros: {', '.join(selected_genres)}")
                for index, row in recommendations.iterrows():
                    st.write(f"{row['title_pt']} - Rating: {row['rating_original']} - Ano: {row['year_original']}")
            else:
                st.write("Por favor, selecione pelo menos um gênero.")

# Interface organizada com colunas e containers
col1, col2 = st.columns([1, 3])
with col1:
    st.sidebar.image("LogoNetflix.png", use_column_width=True)
    st.sidebar.title("Configurações de Recomendação")
    method = st.sidebar.selectbox("Método de Recomendação", ["Método 1: Escolha de Sinopse", "Método 2: Escreva sua Própria Sinopse (Cluster)", "Método 3: Escreva sua Própria Sinopse (Proximidade)", "Método 4: Escolha por Gênero"])

with col2:
    st.markdown("<h1 style='color: #FF6347; font-size: 36px;'>Recomendação de Filmes por Clusterização</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: gray; font-size: 20px;'>Encontre filmes incríveis baseados em sinopses ou gêneros favoritos!</h4>", unsafe_allow_html=True)

    # Exibir o método de recomendação
    handle_recommendation(method)

# Estilização dos botões com bordas arredondadas, cores e sombras
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #FF6347;
        color: white;
        border-radius: 12px;
        width: 100%;
        height: 45px;
        font-size: 18px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)