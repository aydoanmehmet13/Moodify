# 🎵 Spotify Music Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

*A collaborative content-based recommendation system powered by audio feature analysis*

**A team project by 4 Computer Science & Engineering students**

</div>

---

## 📋 Overview

This project implements a **content-based recommendation system** that analyzes Spotify's audio features to suggest similar songs. Developed as a collaborative effort by a team of 4 students, the system processes over **114,000 tracks** and uses Euclidean distance calculations to deliver personalized music recommendations.

---

## 🎯 Key Highlights

- 🔍 **Large-Scale Data Processing**: Cleaned and analyzed 114,000+ tracks
- 🎼 **Custom Algorithm**: Implemented recommendation engine using Euclidean distance metrics
- 🇹🇷 **Turkish Music Detection**: Specialized identification system for Turkish music
- 📊 **Exploratory Data Analysis**: Deep analysis of energy, tempo, and audio trends
- 👥 **Collaborative Development**: Built by 4-person student team
- ⚡ **Efficient Pipeline**: Optimized data processing with Pandas and NumPy

---

## 👥 Team & Contributions

This project was developed collaboratively by a 4-person team of engineering students.

### Project Lead & Primary Contributions
**Mehmet Aydoğan** - *Lead Data Researcher & Analyst*
- Led data cleaning and preprocessing pipeline (114K+ tracks)
- Implemented Euclidean distance recommendation algorithm
- Developed Turkish music identification system
- Conducted exploratory data analysis and visualization
- Project documentation and repository management

### Team Collaboration
The project benefited from collaborative efforts in:
- Dataset research and acquisition
- Feature engineering and selection
- Algorithm testing and validation
- Code review and optimization

---

## ✨ Features

### Core Functionality
- **Content-Based Filtering**: Recommendations based on audio features (tempo, energy, danceability, acousticness, etc.)
- **Multi-Feature Analysis**: Uses 13+ audio attributes for similarity calculations
- **Turkish Music Classifier**: Identifies Turkish tracks using metadata analysis
- **Scalable Processing**: Handles large datasets efficiently

### Technical Implementation
- Comprehensive data cleaning and preprocessing
- Missing value handling and outlier detection
- Feature normalization and scaling
- Custom Euclidean distance implementation
- Exploratory Data Analysis with visualizations

---

## 🛠️ Tech Stack

**Programming:** Python 3.8+  
**Data Processing:** Pandas, NumPy, SciPy  
**Visualization:** Matplotlib, Seaborn  
**Development:** Jupyter Notebook  
**Version Control:** Git & GitHub

---

## 📊 Dataset

The project uses the **Spotify Tracks Dataset** containing:

- **Size**: 114,000+ tracks
- **Features**: 
  - Audio characteristics (tempo, energy, danceability, loudness, speechiness)
  - Metadata (artist, album, release date, popularity)
  - Musical properties (key, mode, time signature)

### Data Preprocessing

1. **Data Cleaning**: Removed duplicates, handled missing values, filtered invalid data
2. **Feature Engineering**: Normalized features (0-1 scale), created composite metrics
3. **Quality Assurance**: Validated data integrity and consistency

---

## 🔬 Methodology

### Euclidean Distance-Based Recommendation

The recommendation engine measures similarity between songs using **Euclidean distance** across multiple audio features:

```python
# Distance calculation for multi-dimensional feature space
distance = sqrt(Σ(feature_i_song1 - feature_i_song2)²)
```

**Implementation Details:**
- Calculated distances across 8-10 key audio features
- Normalized all features to 0-1 range before calculation
- Used NumPy vectorization for efficient computation on 114K tracks
- Leveraged scipy.spatial.distance for optimized calculations

**Why Euclidean Distance?**
- ✅ Simple and interpretable metric
- ✅ Computationally efficient for large datasets
- ✅ Works well with normalized numerical features
- ✅ Captures overall similarity across multiple dimensions
- ✅ Proven effectiveness in content-based filtering

**Alternative Considered:**
- Cosine Similarity was evaluated but Euclidean distance performed better for our normalized multi-feature dataset

### Recommendation Process

1. **Feature Extraction** → Extract normalized audio features from input track
2. **Distance Calculation** → Compute Euclidean distance to all 114K tracks
3. **Ranking** → Sort by ascending distance (closest = most similar)
4. **Filtering** → Apply optional filters (genre, language, year)
5. **Output** → Return top N recommendations

---

## 📈 Results & Insights

This collaborative project successfully demonstrates content-based filtering on a large-scale dataset.

### Key Findings from EDA

**Energy vs Tempo Correlation**
- Strong positive correlation between energy and tempo
- High-energy tracks typically have faster BPM
- Useful insight for mood-based recommendations

**Genre Distribution**
- Pop and Hip-Hop dominate the dataset
- Good genre diversity enables varied recommendations
- Balanced representation across multiple genres

**Turkish Music Characteristics**
- Successfully identified Turkish tracks using metadata
- Distinct audio patterns observed in Turkish music
- Enables culture-specific recommendations

### Project Outcomes

✅ **Data Processing**: Successfully cleaned and processed 114,000+ tracks  
✅ **Algorithm Implementation**: Built custom Euclidean distance recommendation engine  
✅ **Feature Analysis**: Comprehensive EDA revealing audio patterns and trends  
✅ **Turkish Music Detection**: Specialized system for identifying Turkish tracks  
✅ **Team Collaboration**: Successfully coordinated 4-person development team

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended

### Quick Start

```bash
# Clone the repository
git clone https://github.com/aydoanmehmet13/Spotify-Recommendation-System.git
cd Spotify-Recommendation-System

# Install dependencies
pip install pandas numpy scipy matplotlib seaborn jupyter

# Launch Jupyter Notebook
jupyter notebook
```

### Basic Usage

```python
# Load your dataset
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

df = pd.read_csv('data/spotify_tracks.csv')

# Get recommendations for a track
def get_recommendations(track_id, n=10):
    # Extract features of input track
    # Calculate Euclidean distances to all tracks
    # Return top N similar tracks
    pass

recommendations = get_recommendations('your_track_id', n=10)
print(recommendations)
```

---

## 📂 Project Structure

```
Spotify-Recommendation-System/
│
├── data/                    # Dataset files
├── notebooks/               # Jupyter notebooks
│   ├── EDA.ipynb           # Exploratory analysis
│   ├── cleaning.ipynb      # Data preprocessing
│   └── recommendation.ipynb # Main algorithm
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## 🎓 Learning Outcomes

Through this collaborative project, the team gained experience in:

- **Team Collaboration**: Coordinating a 4-person engineering team
- **Data Engineering**: Processing 114K+ records with Pandas
- **Algorithm Design**: Implementing Euclidean distance-based recommendation system
- **Python Programming**: Advanced data manipulation and scientific computing
- **Data Analysis**: Exploratory analysis and statistical visualization
- **Problem Solving**: Handling missing data, outliers, and performance optimization

### Personal Contributions (Lead Role)
As Lead Data Researcher & Analyst, I specifically:
- Architected the data cleaning pipeline for 114K+ tracks
- Implemented the core Euclidean distance algorithm using NumPy/SciPy
- Developed the Turkish music identification system
- Led exploratory data analysis and visualization efforts
- Managed project documentation and GitHub repository

---

## 🔮 Future Enhancements

- [ ] Implement collaborative filtering for hybrid recommendations
- [ ] Add web interface using Streamlit or Flask
- [ ] Integrate Spotify API for real-time data
- [ ] Experiment with machine learning models (K-NN, Neural Networks)
- [ ] Compare Euclidean distance with Cosine Similarity
- [ ] Deploy as REST API

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Project Lead

**Mehmet Aydoğan** - *Lead Data Researcher & Analyst*

🎓 Electrical & Electronics Engineering Student @ İzmir Democracy University  
🔗 LinkedIn: [linkedin.com/in/mehmet-aydoganEE](https://linkedin.com/in/mehmet-aydoganEE)  
📧 Email: aydoanmehmet13@gmail.com  
💻 GitHub: [@aydoanmehmet13](https://github.com/aydoanmehmet13)

---

## 🙏 Acknowledgments

- Collaborative team of 4 engineering students
- Spotify for providing comprehensive audio feature data
- İzmir Democracy University for academic support
- The open-source community for tools and inspiration

---

## 📚 References

- [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api/)
- [Content-Based Filtering](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)
- [Euclidean Distance in Machine Learning](https://en.wikipedia.org/wiki/Euclidean_distance)
- [SciPy Spatial Distance Metrics](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

*A collaborative student project built with ❤️ and 🎵*

</div>
