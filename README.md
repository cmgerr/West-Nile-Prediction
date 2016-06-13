# WestNilePrediction

## Problem Statement

Given historical observations on the appearance of West Nile Virus in the Chicago area, as well as historical weather data, can we predict when and where the virus is likely to appear? Our objective is to identify areas likely to see West Nile Virus cases in order to proactively spray those locations to prevent the spread of the disease.

## Project Roadmap

Initial steps:

1. Clean data

    a) for train.csv, specifically combine rows that represent the same testing day. Due to a quirk in the testing methodology, the number of mosquitos is capped at 50, so if there were greater than 50 mosquitos tested this would be represented as more than one observation
    
    b) for weather.csv, explore data to determine best way to merge it with the train.csv set (based on time and location)

2. Preliminary EDA

    a) train.csv alone
    
    b) weather alone
    
    c) train.csv merged with weather data
