import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# Set halaman konfigurasi
st.set_page_config(page_title="Sistem Rekomendasi Webtoon", layout="wide")

# Fungsi membersihkan teks
clean_spcl = re.compile(r'[/(){}\[\]\|@,;]')
clean_symbol = re.compile(r'[^0-9a-z #+_]')
sastrawi = StopWordRemoverFactory()
stopworda = sastrawi.get_stop_words()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = text.lower()
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub('', text)
    text = stemmer.stem(text)
    text = ' '.join(word for word in text.split() if word not in stopworda)
    return text

# Membaca dataset
@st.cache_data
def load_data():
    webtoon_df = pd.read_excel('webtoon-scraper.xlsx')
    webtoon_df = webtoon_df[webtoon_df['judul'].notnull()]
    webtoon_df['desc_clean'] = webtoon_df['judul'].apply(clean_text)
    webtoon_df.set_index('judul', inplace=True)
    return webtoon_df



webtoon_df = load_data()

# Membuat TF-IDF dan Cosine Similarity
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0)
tfidf_matrix = tf.fit_transform(webtoon_df['desc_clean'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(webtoon_df.index)

def recommendations(judul, top=10):
    recommended_webtoon = []
    matching_indices = indices[indices.str.contains(judul, case=False, na=False)]
    if matching_indices.empty:
        return ["Webtoon tidak ditemukan. Coba judul lain."]
    idx = matching_indices.index[0]
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
    top = top + 1
    top_indexes = list(score_series.iloc[0:top].index)
    for i in top_indexes:
        recommended_webtoon.append(list(webtoon_df.index)[i] + " - Skor: {:.2f}".format(score_series[i]))
    return recommended_webtoon

# Tampilan aplikasi
st.title("✨ Sistem Rekomendasi Webtoon 📚")

st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("Aplikasi ini membantu Anda menemukan rekomendasi webtoon berdasarkan judul yang Anda masukkan. 🚀")

st.sidebar.markdown("📊 *Fitur Utama:*")
st.sidebar.markdown("- *Input Judul*")
st.sidebar.markdown("- *Rekomendasi Terbaik*")

webtoon = st.text_input("Masukkan Judul Webtoon", placeholder="Misal: 'True Beauty'")
rekomendasi = st.button("🎯 Cari Rekomendasi")

if rekomendasi:
    with st.spinner("🔎 Mencari rekomendasi..."):
        hasil = recommendations(webtoon, 10)
        st.subheader("🎉 Rekomendasi untuk Anda:")
        for item in hasil:
            st.markdown(f"🔹 {item}")

# Tambahkan footer
st.markdown("---")
st.markdown("22.12.2645 Rizki Abdullah dan 22.12.2663 Putri Shania")
