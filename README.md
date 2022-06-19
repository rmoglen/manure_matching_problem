# Manure Matching Problem
This project find the least-cost redistribution of manure across the geographic region of interest, based on the [NuGIS database](http://nugis.ipni.net/). 

The current status of the project is summarized in `doc\INFEWS_course_presentation` as of 6/18/2022

## Data
+ [County Supply/ Demand](http://nugis.ipni.net/): We used the "Balance" columns for the year 2016
+ [County Distances](https://www.nber.org/research/data/county-distance-database): Straight-line distances. Note that this file was too big to push to github. To run code, download the file to the `data` directory as `sf12010countydistancemiles.csv`
+ [Cost of manure transport](https://www.bts.gov/content/average-freight-revenue-ton-mile): $0.0089 / mi - lb N
+ [Variable cost of applying manure (price of manure)](https://www.reuters.com/world/us/us-manure-is-hot-commodity-amid-commercial-fertilizer-shortage-2022-04-06/): $0.63/lb N
+ [Fixed cost of manure disposal](https://ageconsearch.umn.edu/record/24057/): $141,894
+ [Variable cost of manure disposal](https://www.iastatedigitalpress.com/air/article/6256/galley/6121/download/): $0.53/lb N
+ [Variable cost of synthetic fertilizer](https://farmdocdaily.illinois.edu/2022/04/nitrogen-fertilizer-prices-and-supply-in-light-of-the-ukraine-russia-conflict.html): $0.92/lb N

## Running the Model
+ Download straight-line distance between counties as defined above
+ Run `code/manure_matching.py`
  + Select nitrogen or phosphorous as the nutrient of interest by modifying `nutrient` in  `manure_matching.py`
  + Select geographic region of interest by modifying `state` in `manure_matching.py`
+ All figures will automatically save to the `fig\` directory


