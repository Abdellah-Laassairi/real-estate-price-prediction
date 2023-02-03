# Data Challenge 2022

This is the project repository for the 2022 Data Challenge as part of (APST) Statistical Learning course at Ecole Centrale de Nantes and University of Nantes. The final project report is avaialble [at the following link]().

## Installation 
All the projects requirements and libraries are saved in requirements.txt
```bash
pip install -r requirements.txt
```

## Description
Institut Louis Bachelier (ILB) is a sponsored research network in Economics and Finance. It is an association as defined by the law of 1901 and was created in 2008 at the instigation of the Treasury and Caisse des Dépôts et Consignations. 

The ILB Datalab is a team of data scientists working alongside researchers of the ILB network on applied research projects for both public and private actors of our economic and financial ecosystem. The ILB datalab recently collected an extensive amount of French real estate data and would like to conduct analyses and experiments with it. 

## Challenge goals

The project is a regression task that deals with real estate price estimation. Estimating housing real estate price is quite a common topic, with an important litterature on estimating prices based on usual data such as: location, surface, land, number of bedrooms, age of the building... The approaches are usually sufficient to estimate the price range but lack precision. However, few have worked to see if adding photos of the asset would bring complementary information, enabling a more precise price estimation.

The objective is thus to work on modelling French housing real estate prices based on usual hierarchical tabular data and, a few photos (between 1 and 6) for each asset and see if it allows better performance than a model trained without the photos.

The inference pipeline looks as follows :
![](docs/inference.png)