# Hack4Dev-

#Project disclaimer
- This project is solesly for the development of the Hack4Dev project, which is a hackathon project that is being developed by the Hack4Dev team.
- This project is based on the Hack4Dev platform, which is a hackathon platform that is being developed by the PIxelArch team, which is a team of students from the Sol Plaatje Universtiy, any modification or contribution should be permitted by the team and written approval and akcnowledgement of the primary builders of the platform.
- The core challenge is classifying building ages (4 classes) using Landsat spectral reflectance values (Blue, Green, Red, NIR, SWIR1, SWIR2).
- The model is designed to be **transferable**, trained on Madrid data and adapted to Amsterdam using few-shot learning (as few as 5-200 samples per class).
- Spectral signatures of materials like glass, timber, concrete, and brick are captured through annual time-series observations from Landsat 5, 7, 8, and 9.

# Project Roles 
The project roles are as follows:
- Project Manager (Exccellent, https://github.com/de_night_shepherd)
- Researcher (Exccellent, https://github.com/de_night_shepherd)
- Software Developer (Exccellent, https://github.com/de_night_shepherd , )
- Data Scientist 
- UI/UX Designer (Lethabo(https://github.com/lethabo-m954), https://github.com/lethabo-m954)
- Content Writer (https://github.com/de_night_shepherd, https://github.com/de_night_shepherd)
**------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**
# Project Overview
This project is a **Transferable Unit Model** designed for the **Building Age Classification hackathon**. The goal is to classify 30m x 30m pixels into 4 age classes based on their spectral fingerprint. The system is engineered to be city-agnostic, allowing seamless transfer from Mediterranean to Maritime urban environments.

**------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**
### Key Features
 - **Thermal Intelligence:** Maps heat emission levels and Land Surface Temperature (LST) proxies to determine building insulation efficiency.
- **Proximity Analytics:** Real-time distance mapping to urban hubs, transit, and green spaces, enriching the spatial context.
- **Digital Twin Engine:** Acts as a high-fidelity backend that streams real-time building metadata (age, materials, net worth, morphology) to an interactive frontend.
- **Interactive Urban Intelligence:** Integrated with map-based frontends to provide "X-Ray" building insights upon zooming.
- **Spatial-Temporal Transformer (STT):** Uses a Multi-head Attention mechanism to scan 40-year spectral trajectories, pinpointing exact construction events.
- **Graph Neural Morphology:** Learns the "urban fabric" by treating buildings as nodes in a graph, capturing topological growth patterns.
- **Bayesian Uncertainty (MC-Dropout):** Provides confidence scores for every prediction, enabling intelligent active learning and deployment safety.
- **Metric Learning Unit:** Uses a Deep Embedding Network trained with **Triplet Loss** to learn city-agnostic age representations.
- **Metaheuristic Optimization:** Employs **Particle Swarm Optimization (PSO)** and **Memetic Algorithms** to fine-tune the master architecture for the ultimate **Macro F1-Score**.

**------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**
### Project Structure
The implementation is divided into four main Jupyter Notebooks:
1.  **[UnitModel_1_Intro.ipynb](file:///home/de_night_shepherd/Hack4Dev/UnitModel_1_Intro.ipynb):** Challenge overview, class definitions (1-4), and transfer strategy.
2.  **[UnitModel_2_Data.ipynb](file:///home/de_night_shepherd/Hack4Dev/UnitModel_2_Data.ipynb):** Loading Parquet datasets for Madrid and Amsterdam.
3.  **[UnitModel_3_Preprocessing.ipynb](file:///home/de_night_shepherd/Hack4Dev/UnitModel_3_Preprocessing.ipynb):** Class derivation from `weighted_mean_year` and spectral feature engineering.
4.  **[UnitModel_4_Implementation.ipynb](file:///home/de_night_shepherd/Hack4Dev/UnitModel_4_Implementation.ipynb):** Transfer procedure and PSO-driven F1 optimization.

**------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**
### Getting Started
To run the project, ensure you have the following libraries installed:
```bash
pip install pandas pyarrow geopandas scikit-learn joblib numpy
```
**------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**
# Project Metrics and Parameters
The primary evaluation metric is the **Macro F1-Score**. 
We make use of **PSO** to ensure that every hyperparameter is optimized. Each "particle" explores the parameter space to find the best configuration for the specific city context.
The **Memetic Algorithm** integration ensures robustness, catering to live metadata ingested directly from satellite spectral imagery regardless of the training location.

**------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

# Data & APIs
- **Landsat Data:** Annual observations from 1984–2024, harmonized to the OLI scale.
- **Parquet Format:** Data is stored in high-performance Parquet files (`madrid_train.parquet`, `amsterdam_transfer.parquet`).
- **Spectral Features:** Captures reflectance in visible, NIR, and SWIR bands to determine material age and urban density.

**------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**
#  Model turning
- **Transferable Architecture:** A few-shot learning strategy that adapts a Madrid-trained model to Amsterdam using **Prototype Matching**.
- **Metaheuristic Optimization:** Employs **Particle Swarm Optimization (PSO)** and **Memetic Algorithms** to fine-tune the embedding space and maximize the **Macro F1-Score**.
- **Metric Learning Unit:** Uses a Deep Embedding Network trained with **Triplet Loss** to learn city-agnostic age representations.
- **Spectral Intelligence:** Leverages 6 Landsat bands and engineered indices (NDVI, NDBI, NDWI) for city-agnostic spectral signatures.
- **Model requests:** We given the tabular data of the pixel, we predict the age class of the pixel.