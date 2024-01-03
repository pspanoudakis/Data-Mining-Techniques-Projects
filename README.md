# Data Mining Techniques Projects
This is a series of projects for the Spring 2023 **Data Mining Techniques** course on [DIT@UoA](https://www.di.uoa.gr/en).

## Project 1 - Customer Personality Analysis
Given a dataset which describes the customers of a company, we try to draw deductions on 
- The profile of the customers who are more likely to spend more
- The campaign channels which bring more revenue
- The purchase channels which bring more revenue

To reach such conclusions, we use common data mining techniques:
- Data **preprocessing & cleaning**
- **Generation** of new data features using given ones
- Elimination of **outliers**
- Data **Visualization**, e.g. using heatmaps, histograms and bar plots
- **Principal Component Analysis**, to reduce the number of features of the data to extract clusters from
- Cluster extraction using **Agglomerative Clustering** & **K-Means**

## Project 2 - Book Recommendation & Classification
Given a [Goodreads](https://www.goodreads.com/) books dataset:
- We visualize our data and extract deductions using the techniques mentioned in project 1. We also emphasize on extensive Pandas `DataFrame` manipulation, to collect various metrics and statistics on our data.
- We develop a **Book Recommendation System** which can recommend similar dataset books given a specific book id:
    - We vectorize the description of each book, using **TF-IDF**
    - The recommender caclulates the **cosine similarity** for all book descriptions in an efficient way (see [`Pairwise Calculator`](https://github.com/pspanoudakis/Data-Mining-Techniques-Projects/blob/master/project2/modules/recommender.py#L19))
    - We can then query the recommender to return the most similar books for the given one
- We develop a **Book Genre Classifier**, which estimates the Genre for a book given the description of it:
    - We vectorize each description using the mean of the included [**Word2Vec**](https://radimrehurek.com/gensim/models/word2vec.html) vectors, to create the training & test data
    - We use an scikit-learn base classifier such as **Naive Bayes**, **Random Forest** and **Support Vector Classifier**  to perform **K-Fold Cross-Validation**, calculate metrics (accuracy, f-score, precision & recall) and measure the performance of our classifier.

## Technologies & Tools used for development
- **Pandas** & **NumPy**
- **matplotlib** & **seaborn**
- **scikit-learn**
- VS Code & Google Colab

## Repository content
Both projects include the following:
- `hw[x].pdf` file describing the corresponding project tasks in detail
- `.ipynb` and `.py` implementation files
