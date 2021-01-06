# spread_of_misinformation

### Running The Project
- To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`.

### Using `run.py`
- To get the data, from the project root directory, run `python run.py data`
    * This downloads the data from the Panacea Lab using the dates specified in `data/dates.txt` in the directory specified in `config/data-params.json`.
    * The dehydrated tweets are stored in `data/raw`.
    * The tweet IDs we select to rehydrate are then stored in `data/temp` along with the actual tweets themselves once they are rehydrated.

- To get a cleaned version of the data for EDA, from the project root directory, run `python run.py clean`
    * This takes the rehydrated tweets obtained from running `python run.py data` and creates a csv file with the necessary features of interest specified in `config/clean-params.json`.
    * The resulting csv, `clean_tweets.csv`, is stored in `data/temp`.

- To run the EDA, from the project root directory, run `python run.py eda`
    * This takes the scientific and conspiracy hashtags specified in `eda-params.json` and plots the usage of these various hashtags over time along with a distribution of posts per user and a csv containing the top fifty most commonly used hashtags.
    * These images and the csv are stored in `data/report`.
    * In addition, this runs the notebook called `eda_report.ipynb` which is stored in `notebooks`. A resulting html file is generated and stored in `report/eda_report.html` that visializes the images and csv file in Markdown.
    
    
 ### Responsibilities
