# Agriview

## The intersection of computer vision and LLM's. A chatbot designed to convey information on fields using semantic segmentation on satelite imagery. 

Agriview is an agricultural intelligence platform that leverages satellite imagery, computer vision, and large language models to provide actionable insights for farmers, traders or land surveyors.

### Problem Statement

Agricultural decision-making requires complex data analysis from satellite imagery. Agriview simplifies this process by translating complex satellite data into understandable, actionable insights.


### Technologies

 - Satellite Data Acquisition: Copernicus Browser (Sentinel-2)
 - Segmentation model: Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks (https://github.com/VSainteuf/utae-paps?tab=readme-ov-file)
 - Image Processing: Rasterio
 - AI Interaction: LlamaIndex
 - Computer Vision: Custom crop field segmentation model
 - Language Model: LLAMA 8B Instruct

### Key Features
- Intelligent Data Parsing
- Converts plain text instructions into computational variables
- Uses LLAMA 8B Instruct model for precise instruction parsing

#### Satellite Image Analysis

- Segments crop fields from satellite time series data
- Identifies and masks different crop types
- Calculates agricultural metrics (e.g., moisture index)

#### Retrieval-Augmented Generation (RAG)

- Stores and references previous query information
- Enables multi-year comparative analysis

### Workflow

- Input: User provides a natural language query
- Parsing: LLAMA model transforms query into actionable variables
- Image Processing: Computer vision model analyzes satellite imagery
- Metric Calculation: Extracts relevant agricultural metrics
- Insight Generation: LLM provides contextualized, actionable insights

#### Example Use Case
Query: "What's the agricultural outlook for wheat in Kentucky this season?"

Response: "The average moisture NDMI of wheat fields at this time of year in Kentucky is [X], indicating a high yield compared to previous years where it was [Y]."


### Future Roadmap

 - Enhanced segmentation models
 - Expanded crop and region coverage
 - More advanced predictive analytics
 - Time series data for historic analysis 

