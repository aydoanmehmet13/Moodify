import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from math import pi

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify AI Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    file_path = "dataset.csv"

    # Check if file exists locally
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # Remove unnecessary index column if present
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)
            return df
        except Exception as e:
            st.error(f"Error loading local file: {e}")
            return pd.DataFrame()
    else:
        # If file is missing, provide upload option
        st.warning("⚠️ 'dataset.csv' not found. Please upload the dataset.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)
            return df
        return pd.DataFrame()

# Load the dataset
df = load_data()

# App Header
st.title("🎵 Spotify Analysis & AI Recommendation System")
st.markdown("""
This application leverages **Machine Learning techniques** and **Statistical Analysis** to provide:
1.  **Deep Exploratory Data Analysis (EDA)** of music features.
2.  **AI-Powered Song Recommendations** based on user mood and audio features.
""")

# Stop execution if data is not loaded
if df.empty:
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["📊 Visual Analysis (EDA)", "🎧 AI Recommendation Engine"])
st.sidebar.info("Developed for Data Science Presentation")

# -----------------------------------------------------------------------------
# MODULE 1: EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
if menu == "📊 Visual Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Analyzing audio features, genre distributions, and correlations.")

    # -- Data Preparation for Turkish Songs --
    mask_genre = df['track_genre'] == 'turkish'
    # Regex to identify Turkish characters in names
    tr_chars = r'[ğşİı]'
    mask_text = df['track_name'].astype(str).apply(lambda x: bool(re.search(tr_chars, x))) | \
                df['artists'].astype(str).apply(lambda x: bool(re.search(tr_chars, x)))
    turkish_songs = df[mask_genre | mask_text]

    # -- Row 1: Histograms --
    st.subheader("1. Distribution of Audio Features")
    audio_features = ['danceability', 'energy', 'valence', 'tempo', 'popularity', 'acousticness']

    fig1, ax1 = plt.subplots(2, 3, figsize=(15, 10))
    ax1 = ax1.flatten()
    for i, col in enumerate(audio_features):
        if col in df.columns:
            sns.histplot(df[col], bins=30, kde=True, color="#1DB954", edgecolor="black", ax=ax1[i])
            ax1[i].set_title(f"{col.capitalize()} Distribution")
            ax1[i].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

    # -- Row 2: Heatmap & Genres --
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2. Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax2)
        st.pyplot(fig2)

    with col2:
        st.subheader("3. Top 10 Popular Genres")
        if 'track_genre' in df.columns and 'popularity' in df.columns:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            top_genres = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10)
            sns.barplot(x=top_genres.values, y=top_genres.index, palette="magma", ax=ax3)
            ax3.set_xlabel("Average Popularity Score")
            st.pyplot(fig3)

    # -- Row 3: Turkish Analysis --
    st.subheader("4. Genre Distribution of Turkish Songs")
    if not turkish_songs.empty:
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        genre_counts = turkish_songs['track_genre'].value_counts().head(10)
        sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="Reds_r", ax=ax4)
        for i, v in enumerate(genre_counts.values):
            ax4.text(v + 1, i, str(v), color='black', va='center', fontweight='bold')
        st.pyplot(fig4)
    else:
        st.info("No specific Turkish songs found in the dataset.")

    # -- Row 4: Advanced Charts (Radar & Scatter) --
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("5. Radar Chart: Turkish vs Global")
        if not turkish_songs.empty:
            radar_feats = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'liveness']
            tr_means = turkish_songs[radar_feats].mean().tolist()
            gl_means = df[radar_feats].mean().tolist()
            tr_means += tr_means[:1]
            gl_means += gl_means[:1]
            N = len(radar_feats)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            fig5 = plt.figure(figsize=(6, 6))
            ax5 = plt.subplot(111, polar=True)
            ax5.plot(angles, tr_means, linewidth=2, linestyle='solid', label='Turkish Avg', color='red')
            ax5.fill(angles, tr_means, 'red', alpha=0.25)
            ax5.plot(angles, gl_means, linewidth=2, linestyle='solid', label='Global Avg', color='gray')
            ax5.fill(angles, gl_means, 'gray', alpha=0.1)
            plt.xticks(angles[:-1], [f.capitalize() for f in radar_feats])
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            st.pyplot(fig5)

    with col4:
        st.subheader("6. Mood Map (Energy vs Valence)")
        fig6, ax6 = plt.subplots(figsize=(8, 8))
        sample_global = df.sample(n=2000, random_state=42)
        ax6.scatter(sample_global['valence'], sample_global['energy'], c='lightgray', alpha=0.5, label='Global Sample', s=15)
        if not turkish_songs.empty:
            ax6.scatter(turkish_songs['valence'], turkish_songs['energy'], c='red', alpha=0.7, label='Turkish Songs', s=20)
        ax6.axhline(0.5, color='black', linestyle='--', alpha=0.3)
        ax6.axvline(0.5, color='black', linestyle='--', alpha=0.3)
        ax6.text(0.95, 0.95, 'HAPPY / ENERGETIC', ha='right', va='top', fontsize=9, color='blue')
        ax6.text(0.05, 0.95, 'ANGRY / TENSE', ha='left', va='top', fontsize=9, color='purple')
        ax6.text(0.95, 0.05, 'CALM / PEACEFUL', ha='right', va='bottom', fontsize=9, color='green')
        ax6.text(0.05, 0.05, 'SAD / DEPRESSING', ha='left', va='bottom', fontsize=9, color='brown')
        ax6.set_xlabel('Valence (Positivity)')
        ax6.set_ylabel('Energy (Intensity)')
        ax6.legend()
        st.pyplot(fig6)

# -----------------------------------------------------------------------------
# MODULE 2: AI RECOMMENDATION ENGINE
# -----------------------------------------------------------------------------
elif menu == "🎧 AI Recommendation Engine":
    st.header("🎧 AI-Powered Recommendation Engine")
    st.markdown("Filter songs by **Genre**, **Origin**, and **Mood**.")

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        genre_input = st.text_input("Preferred Genre (e.g., pop, jazz, rock)", placeholder="Leave empty for all")
        filter_turkish = st.checkbox("Show only Turkish songs?")
    with col_in2:
        mood_selection = st.selectbox(
            "Select Your Mood:",
            ["Happy (Joyful & Positive)", "Energetic (Fast & Intense)", "Calm (Slow & Acoustic)", "Sad (Melancholic & Slow)"]
        )

    if st.button("Get AI Recommendations"):
        results_df = df.copy()
        if genre_input:
            results_df = results_df[results_df['track_genre'].astype(str).str.contains(genre_input, case=False, na=False)]
        if filter_turkish:
            tr_chars = r'[ğşİı]'
            mask_tr = results_df['track_genre'].str.contains('turkish', case=False) | \
                      results_df['track_name'].astype(str).str.contains(tr_chars) | \
                      results_df['artists'].astype(str).str.contains(tr_chars)
            results_df = results_df[mask_tr]

        if results_df.empty:
            st.error("No songs found matching your initial criteria.")
        else:
            target_features = {'danceability': 0.5, 'energy': 0.5, 'valence': 0.5, 'acousticness': 0.5}
            if "Happy" in mood_selection:
                results_df = results_df[results_df['valence'] > 0.5]
                target_features.update({'danceability': 0.8, 'valence': 0.9, 'energy': 0.7})
            elif "Energetic" in mood_selection:
                results_df = results_df[results_df['tempo'] > 110]
                results_df = results_df[results_df['energy'] > 0.6]
                target_features.update({'energy': 0.9, 'danceability': 0.7, 'acousticness': 0.0})
            elif "Calm" in mood_selection:
                results_df = results_df[results_df['energy'] < 0.6]
                target_features.update({'energy': 0.2, 'acousticness': 0.9, 'danceability': 0.3})
            elif "Sad" in mood_selection:
                results_df = results_df[results_df['valence'] < 0.5]
                target_features.update({'valence': 0.1, 'energy': 0.2, 'danceability': 0.2})

            if results_df.empty:
                st.warning("⚠️ No songs matched the strict mood criteria.")
            else:
                cols_to_use = ['danceability', 'energy', 'valence', 'acousticness']
                results_df = results_df.dropna(subset=cols_to_use)
                results_df['similarity_score'] = 0
                for col in cols_to_use:
                    results_df['similarity_score'] += (results_df[col] - target_features[col]) ** 2

                candidate_pool = results_df.sort_values('similarity_score', ascending=True).head(50)
                final_recommendations = candidate_pool if len(candidate_pool) < 5 else candidate_pool.sample(n=5)

                st.success(f"✅ Recommendations Generated!")
                for _, row in final_recommendations.iterrows():
                    match_pct = max(0, min(100, 100 - (row['similarity_score'] * 10)))
                    spotify_link = f"https://open.spotify.com/track/{row['track_id']}"
                    col_img, col_info = st.columns([1, 4])
                    with col_info:
                        st.markdown(f"### 🎵 {row['track_name']}")
                        st.markdown(f"**Artist:** {row['artists']}")
                        st.markdown(f"**Genre:** `{row['track_genre']}` | **Match:** `{match_pct:.1f}%`")
                        st.markdown(f"[Listen on Spotify]({spotify_link})")
                    st.markdown("---")