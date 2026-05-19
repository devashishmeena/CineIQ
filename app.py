import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.recommender import CineIQRecommender

# Page Configuration
st.set_page_config(
    page_title="CineIQ | Next-Gen Recommendations",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Aesthetics
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    .stApp {
        background: radial-gradient(circle at top, #1A1F2C 0%, #0A0C10 100%);
        color: #E2E8F0;
    }

    /* Hero Section */
    .hero-header {
        text-align: center;
        padding-top: 2rem;
        background: linear-gradient(90deg, #FF0055 0%, #7C3AED 50%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.5rem;
        font-weight: 800;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .hero-subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.3rem;
        font-weight: 300;
        margin-top: 1rem;
        margin-bottom: 3.5rem;
        letter-spacing: 0.5px;
    }

    /* Glassmorphism Movie Cards */
    .movie-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        cursor: pointer;
    }
    .movie-card:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Card Elements */
    .movie-rank {
        font-size: 0.9rem;
        font-weight: 800;
        color: #FF0055;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .movie-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #F8FAFC;
        margin: 8px 0 16px 0;
        line-height: 1.3;
    }

    /* Score Badges */
    .score-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }
    .score-pill {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .pill-hybrid {
        background: linear-gradient(135deg, #FF0055 0%, #FF5500 100%);
        color: white;
        box-shadow: 0 4px 10px rgba(255, 0, 85, 0.3);
    }
    .pill-content {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .pill-collab {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F111A;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    return CineIQRecommender()

# Hero Section
st.markdown("<h1 class='hero-header'>CineIQ</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Hyper-Personalized Hybrid Discovery Engine</p>", unsafe_allow_html=True)

with st.spinner("Initializing Neural Pathways..."):
    recommender = load_recommender()

# Sidebar Setup
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3163/3163478.png", width=60)
    st.markdown("### 🎛️ Algorithm Tuning")
    st.markdown("Configure your recommendation parameters.")
    st.write("")
    
    user_id = st.number_input("Target User ID (Memory Matrix)", min_value=1, value=1, step=1)
    
    # Get available movies for dropdown
    movies_df = recommender.movies
    popular_movies = movies_df.dropna(subset=['original_title']).sort_values('popularity', ascending=False).head(1500)
    movie_options = dict(zip(popular_movies['original_title'], popular_movies['id']))
    
    selected_movie_title = st.selectbox("Anchor Movie (What are you in the mood for?)", list(movie_options.keys()))
    top_n = st.slider("Depth of Results", min_value=3, max_value=15, value=8)
    
    st.write("")
    generate_btn = st.button("🚀 Synthesize Matches", type="primary", use_container_width=True)

if generate_btn:
    target_tmdb_id = movie_options[selected_movie_title]
    
    st.markdown(f"### ✨ Top Matches computed for **{selected_movie_title}**")
    st.write("")
    
    with st.spinner("Aligning Content Vectors & Collaborative Signals..."):
        recs = recommender.recommend(user_id=user_id, target_tmdb_id=target_tmdb_id, top_n=top_n)
    
    if not recs:
        st.warning("No recommendations found.")
    else:
        col1, col_space, col2 = st.columns([1.2, 0.1, 1.0])
        
        with col1:
            for rank, r in enumerate(recs, 1):
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-rank">Match #{rank}</div>
                    <div class="movie-title">{r['title']}</div>
                    <div class="score-container">
                        <div class="score-pill pill-hybrid">
                            ★ {r['final_score']:.2f} Match Score
                        </div>
                        <div class="score-pill pill-content">
                            🧬 Content: {r['content_score']:.2f}
                        </div>
                        <div class="score-pill pill-collab">
                            👥 Crowd: {r['collab_score']:.2f}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.02); padding: 25px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #F8FAFC;">🔍 Algorithmic Transparency</h3>
                <p style="color: #94A3B8; font-size: 0.95rem;">This chart breaks down the exact composition of your recommendations. See how much influence came from the movie's thematic DNA (TF-IDF) vs. the collaborative memory of the crowd (SVD Matrix Factorization).</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Prepare data for Plotly
            titles = [r['title'][:18] + "..." if len(r['title']) > 18 else r['title'] for r in recs]
            content_scores = [r['content_score'] for r in recs]
            collab_scores = [r['collab_score'] for r in recs]
            
            fig = go.Figure()
            # Content Layer
            fig.add_trace(go.Bar(
                y=titles,
                x=content_scores,
                name='Content DNA (TF-IDF)',
                orientation='h',
                marker=dict(
                    color='rgba(59, 130, 246, 0.8)',
                    line=dict(color='rgba(59, 130, 246, 1.0)', width=1)
                )
            ))
            # Collaborative Layer
            fig.add_trace(go.Bar(
                y=titles,
                x=collab_scores,
                name='Crowd Matrix (SVD)',
                orientation='h',
                marker=dict(
                    color='rgba(168, 85, 247, 0.8)',
                    line=dict(color='rgba(168, 85, 247, 1.0)', width=1)
                )
            ))
            
            fig.update_layout(
                barmode='stack',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Outfit', color='#94A3B8', size=13),
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(autorange="reversed"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
