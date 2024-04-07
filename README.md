__Runnig the code:__

First, run the ipynb notebook to create the initial processed dataset.

Afterwards, download the csv file that was created (people.csv) - this is the file that the code in the load_data.py uses as an input.

The load_data function creates lables for 5 records at a time and writes it to a csv called people_embedded_and_scored.csv.

When some records are loaded into people_embedded_and_scores.csv, it is possible to run the classifiers_for_data.py file, which uses people_embedded_and_scored.csv as an input and produces the graphs and representatives from each cluster .

