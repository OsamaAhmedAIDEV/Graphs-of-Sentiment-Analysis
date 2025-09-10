import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Page configuration
st.set_page_config(
    page_title="Emotion Analysis Dashboard",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Original JSON data
@st.cache_data
def load_original_data():
    return {
        "emotions": {
            "anger": 0.5909830331802368,
            "anticipation": 0.05819539353251457,
            "disgust": 0.3745267391204834,
            "fear": 0.020751170814037323,
            "joy": 0.9269858002662659,
            "love": 0.15119020640850067,
            "optimism": 0.6522121429443359,
            "pessimism": 0.021883737295866013,
            "sadness": 0.09075716882944107,
            "surprise": 0.03163930028676987,
            "trust": 0.04792511835694313
        },
        "sentiment": {
            "negative": 0.8719632029533386,
            "neutral": 0.112110435962677,
            "positive": 0.015926310792565346
        },
        "tone": {
            "anger": 0.03126896172761917,
            "disgust": 0.07780278474092484,
            "fear": 0.0038883162196725607,
            "joy": 0.847918689250946,
            "neutral": 0.00484048156067729,
            "sadness": 0.03135845437645912,
            "surprise": 0.002922169864177704
        },
        "toxicity": {
            "toxic": 0.08354666084051132,
            "severe_toxic": 0.0001559352094773203,
            "obscene": 0.00225524278357625,
            "threat": 0.0002731348795350641,
            "insult": 0.003933284431695938,
            "identity_hate": 0.00044485763646662235
        }
    }

@st.cache_data
def generate_overall_random_data(num_samples=100):
    """Generate random data for overall analysis structure"""
    data = []
    random.seed(42)  # For reproducible results
    
    for i in range(num_samples):
        # Generate correlated values (realistic relationships)
        sentiment_score = random.uniform(-1, 1)
        tone_score = random.uniform(-1, 1)
        
        # Make emotion somewhat correlated with sentiment and tone
        emotion_pos = random.uniform(0.5, 2.0) if sentiment_score > 0 else random.uniform(0.2, 1.5)
        emotion_neg = random.uniform(0.2, 1.5) if sentiment_score < 0 else random.uniform(0.1, 1.0)
        emotion_score = emotion_pos - emotion_neg + random.uniform(-0.3, 0.3)
        
        toxicity_score = random.uniform(-0.5, 0.2)  # Usually non-toxic
        
        # Final overall is weighted average
        final_score = (sentiment_score * 0.4 + tone_score * 0.3 + 
                      emotion_score * 0.2 + toxicity_score * 0.1)
        
        sample = {
            "sample_id": i+1,
            "sentiment_score": sentiment_score,
            "tone_score": tone_score,
            "emotion_score": emotion_score,
            "toxicity_score": toxicity_score,
            "final_overall": final_score,
            "sentiment_label": "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral",
            "tone_label": "Positive" if tone_score > 0.2 else "Negative" if tone_score < -0.2 else "Neutral",
            "final_label": "Positive" if final_score > 0.1 else "Negative" if final_score < -0.1 else "Neutral"
        }
        data.append(sample)
    
    return pd.DataFrame(data)

def create_emotion_radar_chart(emotions_data):
    """Create radar chart for emotions"""
    categories = list(emotions_data.keys())
    values = list(emotions_data.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Emotions',
        line_color='rgb(255, 107, 107)',
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="🎭 Emotion Profile (Radar Chart)",
        title_x=0.5,
        height=500
    )
    
    return fig

def create_sentiment_pie_chart(sentiment_data):
    """Create pie chart for sentiment"""
    labels = list(sentiment_data.keys())
    values = list(sentiment_data.values())
    colors = ['#FF4444', '#FFAA44', '#44AA44']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=0.3,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title="😊 Sentiment Distribution",
        title_x=0.5,
        height=400
    )
    
    return fig

def create_top_emotions_bar_chart(emotions_data):
    """Create bar chart for top emotions"""
    sorted_emotions = dict(sorted(emotions_data.items(), key=lambda x: x[1], reverse=True))
    top_5 = dict(list(sorted_emotions.items())[:5])
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_5.keys()),
            y=list(top_5.values()),
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            text=[f'{v:.3f}' for v in top_5.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="🔥 Top 5 Tone",
        title_x=0.5,
        xaxis_title="Tone",
        yaxis_title="Intensity",
        height=400
    )
    
    return fig

def create_toxicity_horizontal_bar(toxicity_data):
    """Create horizontal bar chart for toxicity"""
    labels = list(toxicity_data.keys())
    values = list(toxicity_data.values())
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color='rgba(255, 68, 68, 0.6)',
        text=[f'{v:.4f}' for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="⚠️ Toxicity Analysis",
        title_x=0.5,
        xaxis_title="Toxicity Level",
        height=400
    )
    
    return fig

def create_overall_distribution_histogram(df):
    """Create histogram for final overall scores"""
    fig = px.histogram(
        df, 
        x='final_overall', 
        nbins=25,
        title='📊 Final Overall Score Distribution',
        labels={'final_overall': 'Final Score', 'count': 'Frequency'},
        color_discrete_sequence=['lightblue']
    )
    
    # Add mean line
    mean_score = df['final_overall'].mean()
    fig.add_vline(x=mean_score, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_score:.3f}")
    
    fig.update_layout(height=400)
    return fig

def create_scatter_sentiment_tone(df):
    """Create scatter plot for sentiment vs tone correlation"""
    fig = px.scatter(
        df, 
        x='sentiment_score', 
        y='tone_score',
        color='final_label',
        title='🔗 Sentiment vs Tone Correlation',
        labels={
            'sentiment_score': 'Sentiment Score',
            'tone_score': 'Tone Score'
        },
        color_discrete_map={
            'Positive': '#44AA44',
            'Negative': '#FF4444',
            'Neutral': '#FFAA44'
        }
    )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(height=400)
    return fig

def create_trend_line_chart(df):
    """Create trend line chart for final scores over samples"""
    # Calculate moving averages
    df['ma_5'] = df['final_overall'].rolling(window=5, center=True).mean()
    df['ma_20'] = df['final_overall'].rolling(window=20, center=True).mean()
    
    fig = go.Figure()
    
    # Individual scores
    fig.add_trace(go.Scatter(
        x=df['sample_id'], 
        y=df['final_overall'],
        mode='lines',
        name='Individual Scores',
        line=dict(color='lightblue', width=1),
        opacity=0.6
    ))
    
    # Moving averages
    fig.add_trace(go.Scatter(
        x=df['sample_id'], 
        y=df['ma_5'],
        mode='lines',
        name='5-Point MA',
        line=dict(color='orange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['sample_id'], 
        y=df['ma_20'],
        mode='lines',
        name='20-Point MA',
        line=dict(color='red', width=2)
    ))
    
    # Add reference line at zero
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7)
    
    fig.update_layout(
        title='📈 Final Score Trend Over Samples',
        xaxis_title='Sample ID',
        yaxis_title='Final Overall Score',
        height=400
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    corr_cols = ['sentiment_score', 'tone_score', 'emotion_score', 'toxicity_score', 'final_overall']
    corr_matrix = df[corr_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='🔥 Component Correlation Matrix'
    )
    
    fig.update_layout(height=500)
    return fig

def create_final_score_categories(df):
    """Create categorized analysis of final scores"""
    # Define categories
    very_negative = len(df[df['final_overall'] < -0.5])
    negative = len(df[(df['final_overall'] >= -0.5) & (df['final_overall'] < -0.1)])
    neutral = len(df[(df['final_overall'] >= -0.1) & (df['final_overall'] <= 0.1)])
    positive = len(df[(df['final_overall'] > 0.1) & (df['final_overall'] <= 0.5)])
    very_positive = len(df[df['final_overall'] > 0.5])
    
    categories = ['Very Negative\n(<-0.5)', 'Negative\n(-0.5 to -0.1)', 'Neutral\n(-0.1 to 0.1)', 
                 'Positive\n(0.1 to 0.5)', 'Very Positive\n(>0.5)']
    counts = [very_negative, negative, neutral, positive, very_positive]
    colors = ['darkred', 'lightcoral', 'lightyellow', 'lightgreen', 'darkgreen']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            text=[f'{count}<br>({count/len(df)*100:.1f}%)' for count in counts],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='🎯 Score Range Distribution',
        yaxis_title='Number of Samples',
        height=400
    )
    
    return fig

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Emotion Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        ["🎭 Original JSON Analysis", "📈 Overall Scores Analysis", "🎯 Final Score Focus"]
    )
    
    # Load data
    original_data = load_original_data()
    overall_df = generate_overall_random_data(100)
    
    if analysis_type == "🎭 Original JSON Analysis":
        st.markdown('<div class="section-header">Original JSON Data Analysis</div>', unsafe_allow_html=True)
        
        # Display raw data in expander
        with st.expander("📋 View Original JSON Data"):
            st.json(original_data)
        
        # Key Insights
        st.markdown('<div class="section-header">🔍 Key Insights</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dominant_emotion = max(original_data['emotions'], key=original_data['emotions'].get)
            st.metric("🔥 Dominant Emotion", dominant_emotion, 
                     f"{original_data['emotions'][dominant_emotion]:.3f}")
        
        with col2:
            dominant_sentiment = max(original_data['sentiment'], key=original_data['sentiment'].get)
            st.metric("😊 Primary Sentiment", dominant_sentiment,
                     f"{original_data['sentiment'][dominant_sentiment]:.3f}")
        
        with col3:
            dominant_tone = max(original_data['tone'], key=original_data['tone'].get)
            st.metric("🎭 Primary Tone", dominant_tone,
                     f"{original_data['tone'][dominant_tone]:.3f}")
        
        with col4:
            max_toxicity = max(original_data['toxicity'].values())
            toxicity_risk = "HIGH" if max_toxicity > 0.1 else "LOW"
            st.metric("⚠️ Toxicity Risk", toxicity_risk, f"{max_toxicity:.4f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar Chart
            radar_fig = create_emotion_radar_chart(original_data['emotions'])
            st.plotly_chart(radar_fig, use_container_width=True)
            
            # Top Emotions Bar Chart
            bar_fig = create_top_emotions_bar_chart(original_data['emotions'])
            st.plotly_chart(bar_fig, use_container_width=True)
        
        with col2:
            # Sentiment Pie Chart
            pie_fig = create_sentiment_pie_chart(original_data['sentiment'])
            st.plotly_chart(pie_fig, use_container_width=True)
            
            # Toxicity Horizontal Bar
            tox_fig = create_toxicity_horizontal_bar(original_data['toxicity'])
            st.plotly_chart(tox_fig, use_container_width=True)
        
        # Detailed Analysis
        st.markdown('<div class="section-header">📊 Detailed Breakdown</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎭 Emotions", "😊 Sentiment", "🎵 Tone", "⚠️ Toxicity"])
        
        with tab1:
            emotions_df = pd.DataFrame(list(original_data['emotions'].items()), 
                                     columns=['Emotion', 'Score'])
            emotions_df = emotions_df.sort_values('Score', ascending=False)
            st.dataframe(emotions_df, use_container_width=True)
        
        with tab2:
            sentiment_df = pd.DataFrame(list(original_data['sentiment'].items()), 
                                      columns=['Sentiment', 'Score'])
            st.dataframe(sentiment_df, use_container_width=True)
        
        with tab3:
            tone_df = pd.DataFrame(list(original_data['tone'].items()), 
                                 columns=['Tone', 'Score'])
            tone_df = tone_df.sort_values('Score', ascending=False)
            st.dataframe(tone_df, use_container_width=True)
        
        with tab4:
            toxicity_df = pd.DataFrame(list(original_data['toxicity'].items()), 
                                     columns=['Toxicity Type', 'Score'])
            toxicity_df = toxicity_df.sort_values('Score', ascending=False)
            st.dataframe(toxicity_df, use_container_width=True)
    
    elif analysis_type == "📈 Overall Scores Analysis":
        st.markdown('<div class="section-header">Overall Scores Analysis (100 Random Samples)</div>', unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Total Samples", len(overall_df))
        
        with col2:
            st.metric("📈 Mean Final Score", f"{overall_df['final_overall'].mean():.3f}")
        
        with col3:
            st.metric("📊 Std Deviation", f"{overall_df['final_overall'].std():.3f}")
        
        with col4:
            positive_count = len(overall_df[overall_df['final_label'] == 'Positive'])
            st.metric("✅ Positive Samples", f"{positive_count} ({positive_count/len(overall_df)*100:.1f}%)")
        
        with col5:
            negative_count = len(overall_df[overall_df['final_label'] == 'Negative'])
            st.metric("❌ Negative Samples", f"{negative_count} ({negative_count/len(overall_df)*100:.1f}%)")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution Histogram
            hist_fig = create_overall_distribution_histogram(overall_df)
            st.plotly_chart(hist_fig, use_container_width=True)
            
            # Trend Line Chart
            trend_fig = create_trend_line_chart(overall_df.copy())
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Score Categories
            cat_fig = create_final_score_categories(overall_df)
            st.plotly_chart(cat_fig, use_container_width=True)
        
        with col2:
            # Sentiment vs Tone Scatter
            scatter_fig = create_scatter_sentiment_tone(overall_df)
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Correlation Heatmap
            corr_fig = create_correlation_heatmap(overall_df)
            st.plotly_chart(corr_fig, use_container_width=True)
        
        # Data Table
        st.markdown('<div class="section-header">📋 Raw Data Sample</div>', unsafe_allow_html=True)
        st.dataframe(overall_df.head(20), use_container_width=True)
        
        # Download button
        csv = overall_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name="emotion_analysis_data.csv",
            mime="text/csv"
        )
    
    elif analysis_type == "🎯 Final Score Focus":
        st.markdown('<div class="section-header">Final Overall Score - Focused Analysis</div>', unsafe_allow_html=True)
        
        final_scores = overall_df['final_overall']
        
        # Advanced Statistics
        st.markdown('<div class="section-header">📊 Advanced Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Mean", f"{final_scores.mean():.4f}")
            st.metric("📊 Median", f"{final_scores.median():.4f}")
        
        with col2:
            st.metric("📊 Std Dev", f"{final_scores.std():.4f}")
            st.metric("📊 Variance", f"{final_scores.var():.4f}")
        
        with col3:
            st.metric("📊 Skewness", f"{final_scores.skew():.4f}")
            st.metric("📊 Kurtosis", f"{final_scores.kurtosis():.4f}")
        
        with col4:
            st.metric("📊 Min Score", f"{final_scores.min():.4f}")
            st.metric("📊 Max Score", f"{final_scores.max():.4f}")
        
        # Percentiles
        st.markdown('<div class="section-header">📈 Percentiles</div>', unsafe_allow_html=True)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        perc_data = []
        for p in percentiles:
            perc_data.append({
                'Percentile': f'{p}th',
                'Score': f"{np.percentile(final_scores, p):.4f}"
            })
        
        perc_df = pd.DataFrame(perc_data)
        st.dataframe(perc_df, use_container_width=True)
        
        # Enhanced Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced Histogram with Statistics
            fig = px.histogram(overall_df, x='final_overall', nbins=25, 
                             title='📊 Enhanced Score Distribution')
            
            mean_val = final_scores.mean()
            std_val = final_scores.std()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.3f}")
            fig.add_vline(x=mean_val + std_val, line_dash="dot", line_color="orange",
                         annotation_text=f"+1 STD")
            fig.add_vline(x=mean_val - std_val, line_dash="dot", line_color="orange",
                         annotation_text=f"-1 STD")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cumulative Distribution
            sorted_scores = np.sort(final_scores)
            cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sorted_scores,
                y=cumulative,
                mode='lines',
                name='Cumulative Distribution',
                line=dict(color='purple', width=2)
            ))
            
            fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.7)
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.7)
            
            fig.update_layout(
                title='📈 Cumulative Distribution Function',
                xaxis_title='Final Overall Score',
                yaxis_title='Cumulative Probability'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Category Analysis
        cat_fig = create_final_score_categories(overall_df)
        st.plotly_chart(cat_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🧠 Emotion Analysis Dashboard | Develop By OSAMA AHMED</p>
        <p>📊 Interactive visualization of emotional data patterns and insights</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# Additional utility functions for advanced analysis
def export_analysis_report(original_data, overall_df):
    """Export comprehensive analysis report"""
    report = {
        "original_analysis": {
            "dominant_emotion": max(original_data['emotions'], key=original_data['emotions'].get),
            "emotion_scores": original_data['emotions'],
            "sentiment_breakdown": original_data['sentiment'],
            "tone_analysis": original_data['tone'],
            "toxicity_levels": original_data['toxicity']
        },
        "overall_analysis": {
            "total_samples": len(overall_df),
            "final_score_stats": {
                "mean": overall_df['final_overall'].mean(),
                "median": overall_df['final_overall'].median(),
                "std": overall_df['final_overall'].std(),
                "min": overall_df['final_overall'].min(),
                "max": overall_df['final_overall'].max()
            },
            "label_distribution": overall_df['final_label'].value_counts().to_dict()
        }
    }
    return json.dumps(report, indent=2)
