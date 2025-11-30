# Master's Thesis Data Platform

This repository contains the code for the data analysis and visualization part of a Master's Thesis ("gradu"). The project is structured to first download necessary economic data and then perform analysis and visualization.

## Project Structure

- `1_data_download.ipynb`: Jupyter notebook responsible for fetching data from external sources like FRED (Federal Reserve Economic Data). It saves the processed data into CSV files.
- `2_visuals_analysis.ipynb`: Jupyter notebook that loads the downloaded CSV data to generate plots and perform visual analysis.
- `data_csv/`: Directory where the downloaded and processed data is stored (ignored by git).
- `requirements.txt`: List of Python dependencies required to run the project.

## Setup and Usage

1.  **Clone the repository**

2.  **Set up the environment**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Environment Variables**
    This project uses `python-dotenv` to manage secrets. You need to create a `.env` file in the root directory.
    
    The `1_data_download.ipynb` notebook uses `fredapi`, which requires a FRED API key.
    
    Create a `.env` file and add your API key:
    ```
    FRED_API_KEY=your_api_key_here
    ```

4.  **Run the Notebooks**
    - Start with `1_data_download.ipynb` to download the data.
    - Then run `2_visuals_analysis.ipynb` to generate visualizations.

## Dependencies

- pandas
- pandas_datareader
- fredapi
- numpy
- matplotlib
- seaborn
- python-dotenv
