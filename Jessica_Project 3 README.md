<<<<<<< HEAD

# Planner.Pro+ 

### 1. Commpany Background
 
"We plan your dream Holiday ". Planner.Pro+ is a leading provider of personalized holiday itineraries, catering to travellers worldwide who seek exceptional and bespoke experiences.

### 2. Problem statement
 
Even since Covid-19, there has been a noticeable decline in the company's sales. The reopening of Broaden has not led to an improvement in our sales figures. These could due to varies reasons. First, planning a trip has become increasingly convenient and easier with the information or intineraries shared by online content creators. Second, Travelers no longer have to invest time in browsing through and reading numerous itineraries with ChatGPT. With a simple text input such as "Plan a family-oriented trip to Hong Kong for 4 days 3 nights with some hotel recommendations", ChatGPT will plan it out for the users. 

This project aims to leverage on the vast amount of feedback and inquiries received through multiple channels to generate valuable insights. These channels include customer support portals, social media platforms, surveys, or direct communication channels like email or phone calls. Without leveraging this data, the company is missing out on opportunities to gain deeper understanding of customer needs, preferences, pain points, and overall sentiment towards its products or services. The project's goal is to convert the raw feedback and inquiries received by the company into actionable insights. These insights can serve as a guide for decision-making and drive improvements within the organization. 


### 3. Solution

Create a proof of concept by starting with the most renowned theme parks, namely Disney parks and Universal Studios. 
 
The initial step of the project involves training a machine learning model using data sourced from two subreddit posts, Disneyparks and Universalstudios  By leveraging this data, the project aims to develop a model capable of categorizing both existing and future inquiries and feedback into relevant categories. This classification process enables the organization to systematically organize and analyze the vast amount of textual data received from multiple platforms.  
 
Following the model training phase, the project proceeds to conduct exploratory data analysis (EDA) to uncover patterns, trends, and relationships within the feedback and inquiry dataset. By applying statistical and visualization techniques, valuable insights regarding visitor preferences, concerns, satisfaction levels, and emerging topics of interest would be discovered. 
 
The final phase of the project involves sharing the insights gained from the EDA with relevant departments within the organization. Departments such as  trip curation and marketing teams can benefit greatly from the actionable insights derived from clients' feedback and inquiries. For example, the trip curation department can incorporate feedback when curating future trip, and marketing teams can develop targeted campaigns to improve customer engagement. By disseminating these insights across departments, the organization can collaboratively work towards enhancing the overall customer experience and increasing public engagement. 

### 4. Data 
 
Collected 998 and 991 of posts for Disney Parks and Universal Studios respectively. Each post consists of the following information, title, content, data of posts, number of comments and upvote ratio.  The models below have been trained with the combination of the title and context of the posts.
 
### 5. Text cleaning
 
Data Quality Improvement: Text data often contains errors, inconsistencies, and irrelevant content. Cleaning helps ensure that the data is accurate, reliable, and consistent.
 
### 5.1 Noise Reduction
 
Noise in text data can include special characters, HTML tags, punctuation, and other elements such as emoji and extra space that do not contribute to the analysis or modelling goals. Thus, removes those to reduce noise. 

### 5.2 Standardization
 
Text cleaning often includes standardizing text, such as converting all text to lowercase, to ensure consistency and prevent case-related issues from affecting analysis or modelling.
 
### 5.3 Expended Contractions
 
There are so many contractions, such as I'll  and she'd, in the text we type so to expand them we will use the contractions library. For instance, " I've enjoyed myself today in Disneyland. The firework was amazing " to " I have enjoyed myself today in Disneyland. The firework was amazing". This step is essential to remove stop words like "I have", "She have " and etc.
 
### 5.4. Stopword Removal
 
Stopwords are common words like “the,” “and,” or “in” that are often removed during text cleaning because they do not carry significant meaning for many tasks.
 
### 5.5 Lemmatization
 
These techniques reduce words to their root forms, helping to group similar words. Lemmatization are particularly useful for text analysis tasks where word variants should be treated as the same word. Eg Parks and Park
 
### 5.6 Handling Missing Data
 
Not all content of the posts were in text form. Some of them are images. Thus, replaced null value for content with a space instead. 
 
### 5.7 Tokenization
 
Tokenization is a crucial part of text cleaning. It involves breaking text into individual words or tokens, making analyzing or processing text data easier.

### 6. Exploratory Data Analysis (EDA)

### 6.1 Unigram vs Bigram

Two subreddits exhibit a substantial overlap in shared words. However, as the analysis extends to bigrams, the commonality between the subreddits diminishes. This shift may prove advantageous during model training because the vocabulary differ significantly in terms of the combinations and arrangements of words. 
 
 By incorporating bigrams into the training data, the model can capture more context and dependencies between words, leading to a richer representation of the text data. This richer representation allows the model to better distinguish between the two subreddits and learn more discriminative features for classification. 

 ### 6.2 Top 10  Bigram Words with Upvote Ratio More Than or Equal to 0.9

The upvote ratio represents the proportion of upvotes compared to the total number of votes received. For instance, if a post garners 3 upvotes and 1 downvote, its upvote ratio stands at 75% since 3 constitutes 75% of the total 4 votes. 
 
The top 10  bigram words for Disney Parks are first time, Disney resort, animal kingdom, haunted mansion, Tokyo Disney, Hong Kong, magic kingdom, Disney World, Disney Park,   
 
The top 10  bigram words for universal studios are theme park, harry potter, toadstool café, Mario kart, express pas, early access, Studio Hollywood,  super Nintendo, Nintendo World, Universal Studio.   

### 7. Word Embedding Tactic  
 
TF-IDF is a better choice due to its ability to account for the importance of words in a document relative to a collection of documents. It take into account both the term frequency (how often a word appears in a document) and the inverse document frequency (how common or rare a word is across multiple documents). This helps in handling the issue of common words that might occur frequently in many documents but carry little semantic meaning. As both subreddits are related to theme park, the chance of having common words between them would be quite high. Thus TF-IDF is preferred in this scenarios .  
 
In datasets with high dimensionality and sparse features, such as text data, TF-IDF can perform better than CountVectorizer. This is because TF-IDF reduces the impact of common and less informative terms, leading to more effective feature representations. 


### 8. Models Performance


|Model|Train Accuracy Score|Train Accuracy Score|Sensitivity|Specificity|F1|N-Grams|
|---|---|---|---|---|---|---|
|TF-IDF + Support Vector Classification|1.0|0.929648|0.943144|0.916107|0.930693|Bigram|
|TF-IDF + Logr + SVC + NB|1.0|0.929648|0.933110|0.926174|0.930000|Bigram|
|TF-IDF + Multinomial NB|0.999282|0.929648|0.899666|0.959732|0.927586|Bigram|
|TF-IDF + Logistics Regression|1.0|0.929648|0.923077|0.929530|0.926174|Bigram|
|Fast Text|   -  |   -  |0.924623|0.92462|0.924623|Bigram|
|TF-IDF + Random Forest|1.0|0.904523|0.946488|0.862416|0.908507|Bigram|

Based on F1 score, the best model is TF-IDF + Support Vector Classification.

### Fast Text
In the project, I tried using Fast text to train my model. 

FastText is an open-source library developed by Facebook Research for efficient learning of word representations and text classification. It extends the Word2Vec model by incorporating subword information, making it particularly useful for dealing with out-of-vocabulary words. 

While it uses neural network architecture, it is considered a shallow model compared to deep learning architectures like convolutional neural networks (CNNs) or recurrent neural networks (RNNs). However, for this dataset, it did not perform better than most of the models. 

### 9. Model Comparision 

SVC works better in this project because of its property of working well with non-linear relationship data,  effective with high-dimensional data and adaptability to different distributions.  
 
### 9.1 Non - Linear Relationship  
 
The relationship between words and labels can be complex and non-linear, SVMs with appropriate kernels can better model these relationships compared to the linear decision boundaries of logistic regression. 
 
Also, SVC are less sensitive to noise in the data compared to logistic regression. In text classification, where noise can arise from various sources such as misspellings, abbreviations, or inconsistencies in language usage, SVMs can provide better generalization by effectively ignoring noisy data points or outliers.  
 
In text classification, some features (n-grams) may be highly correlated with each other, leading to multicollinearity issues. SVMs are generally less affected by multicollinearity compared to logistic regression, which can lead to more stable and reliable performance. 
 
### 9.2 Effective with High-Dimensional Data 
 
Text classification often involves high-dimensional feature spaces, where each word or n-gram represents a feature. SVMs are known to perform well in high-dimensional spaces  


### 9.3 Adaptability to different distributions 
 
SVMs are non-parametric models and do not make explicit assumptions about the underlying distribution of the data, making them more adaptable to different types of distributions in text data. Naive Bayes, on the other hand, assumes that features follow a specific distribution (e.g., multinomial or Gaussian), which may not always hold true in practice. 


### 10. Improvement 

### 10.1 Scrape more Data to Train and Test The Model 

More data can help in training complex models that have a large number of parameters. Deep learning models, for example, often require large amounts of data to learn intricate patterns effectively.

### 10.2 Remove some highly distinguishable words to generalise the model

Most of the model having a accuracy score of 1. By removing some highly distinguishable words could help to mitigate overfitting and improve the model's ability to generalize to unseen data. By reducing the influence of words that are unique to specific classes or categories, the model can focus on learning more generalizable patterns.

### 10.3 Use cross-validation to evaluate the performance of the model on the training dataset

Cross-validation provides a robust estimate of the model's performance by evaluating it on multiple subsets of the training data. This approach helps in detecting overfitting.

### 12. Recommondation

### 12.1 Marketing Team 

The marketing team can use the top key words above when crafting their marketing materials. Eg “Do and don’t for your first time in Disneyland”, “Is it worth staying at Disney Resort” and etc to engage with their readers. The website content team can also use the top key words to boost the Search Engine Optimization(SEO) score. The higher score, the higher page ranking on the search page leading to higher visibility and increase traffic. 

### 12.2 Trip Curation Team 

The trip curation team can be more detailed when planning, such as what are the must play rides, route to go to play most of the rides, and etc.


### 11. Moving Forward 

By further categorizing these inputs based on sentiment analysis, we can identify patterns and underlying issues affecting our sales performance. For example, if many inquiries express dissatisfaction with the availability of certain services or destinations, we can prioritize addressing these concerns to better meet customer needs and potentially improve sales. Additionally, understanding customer sentiments and preferences can inform strategic decisions, such as adjusting marketing strategies or refining product offerings to better align with market demands. 

### Conclusion 
 
Overall, the project seeks to unlock the value hidden within the company's feedback data, turning it from a passive collection of information into a proactive tool for driving positive change and delivering better outcomes for both the company and its customers. Additionally, understanding customer sentiments and preferences can inform strategic decisions, such as adjusting marketing strategies or refining product offerings to better align with market demands. 

### Citation

Automatic hyperparameter optimization · fasttext. fastText. (n.d.). 
    https://fasttext.cc/docs/en/autotune.html 

Rastogi, K. (2022, November 22). Text cleaning methods in NLP. Analytics Vidhya. 
    https://www.analyticsvidhya.com/blog/2022/01/text-cleaning-methods-in-nlp/ 

=======

# Planner.Pro+ 

### 1. Commpany Background
 
"We plan your dream Holiday ". Planner.Pro+ is a leading provider of personalized holiday itineraries, catering to travellers worldwide who seek exceptional and bespoke experiences.

### 2. Problem statement
 
Even since Covid-19, there has been a noticeable decline in the company's sales. The reopening of Broaden has not led to an improvement in our sales figures. These could due to varies reasons. First, planning a trip has become increasingly convenient and easier with the information or intineraries shared by online content creators. Second, Travelers no longer have to invest time in browsing through and reading numerous itineraries with ChatGPT. With a simple text input such as "Plan a family-oriented trip to Hong Kong for 4 days 3 nights with some hotel recommendations", ChatGPT will plan it out for the users. 

This project aims to leverage on the vast amount of feedback and inquiries received through multiple channels to generate valuable insights. These channels include customer support portals, social media platforms, surveys, or direct communication channels like email or phone calls. Without leveraging this data, the company is missing out on opportunities to gain deeper understanding of customer needs, preferences, pain points, and overall sentiment towards its products or services. The project's goal is to convert the raw feedback and inquiries received by the company into actionable insights. These insights can serve as a guide for decision-making and drive improvements within the organization. 


### 3. Solution

Create a proof of concept by starting with the most renowned theme parks, namely Disney parks and Universal Studios. 
 
The initial step of the project involves training a machine learning model using data sourced from two subreddit posts, Disneyparks and Universalstudios  By leveraging this data, the project aims to develop a model capable of categorizing both existing and future inquiries and feedback into relevant categories. This classification process enables the organization to systematically organize and analyze the vast amount of textual data received from multiple platforms.  
 
Following the model training phase, the project proceeds to conduct exploratory data analysis (EDA) to uncover patterns, trends, and relationships within the feedback and inquiry dataset. By applying statistical and visualization techniques, valuable insights regarding visitor preferences, concerns, satisfaction levels, and emerging topics of interest would be discovered. 
 
The final phase of the project involves sharing the insights gained from the EDA with relevant departments within the organization. Departments such as  trip curation and marketing teams can benefit greatly from the actionable insights derived from clients' feedback and inquiries. For example, the trip curation department can incorporate feedback when curating future trip, and marketing teams can develop targeted campaigns to improve customer engagement. By disseminating these insights across departments, the organization can collaboratively work towards enhancing the overall customer experience and increasing public engagement. 

### 4. Data 
 
Collected 998 and 991 of posts for Disney Parks and Universal Studios respectively. Each post consists of the following information, title, content, data of posts, number of comments and upvote ratio.  The models below have been trained with the combination of the title and context of the posts.
 
### 5. Text cleaning
 
Data Quality Improvement: Text data often contains errors, inconsistencies, and irrelevant content. Cleaning helps ensure that the data is accurate, reliable, and consistent.
 
### 5.1 Noise Reduction
 
Noise in text data can include special characters, HTML tags, punctuation, and other elements such as emoji and extra space that do not contribute to the analysis or modelling goals. Thus, removes those to reduce noise. 

### 5.2 Standardization
 
Text cleaning often includes standardizing text, such as converting all text to lowercase, to ensure consistency and prevent case-related issues from affecting analysis or modelling.
 
### 5.3 Expended Contractions
 
There are so many contractions, such as I'll  and she'd, in the text we type so to expand them we will use the contractions library. For instance, " I've enjoyed myself today in Disneyland. The firework was amazing " to " I have enjoyed myself today in Disneyland. The firework was amazing". This step is essential to remove stop words like "I have", "She have " and etc.
 
### 5.4. Stopword Removal
 
Stopwords are common words like “the,” “and,” or “in” that are often removed during text cleaning because they do not carry significant meaning for many tasks.
 
### 5.5 Lemmatization
 
These techniques reduce words to their root forms, helping to group similar words. Lemmatization are particularly useful for text analysis tasks where word variants should be treated as the same word. Eg Parks and Park
 
### 5.6 Handling Missing Data
 
Not all content of the posts were in text form. Some of them are images. Thus, replaced null value for content with a space instead. 
 
### 5.7 Tokenization
 
Tokenization is a crucial part of text cleaning. It involves breaking text into individual words or tokens, making analyzing or processing text data easier.

### 6. Exploratory Data Analysis (EDA)

### 6.1 Unigram vs Bigram

Two subreddits exhibit a substantial overlap in shared words. However, as the analysis extends to bigrams, the commonality between the subreddits diminishes. This shift may prove advantageous during model training because the vocabulary differ significantly in terms of the combinations and arrangements of words. 
 
 By incorporating bigrams into the training data, the model can capture more context and dependencies between words, leading to a richer representation of the text data. This richer representation allows the model to better distinguish between the two subreddits and learn more discriminative features for classification. 

 ### 6.2 Top 10  Bigram Words with Upvote Ratio More Than or Equal to 0.9

The upvote ratio represents the proportion of upvotes compared to the total number of votes received. For instance, if a post garners 3 upvotes and 1 downvote, its upvote ratio stands at 75% since 3 constitutes 75% of the total 4 votes. 
 
The top 10  bigram words for Disney Parks are first time, Disney resort, animal kingdom, haunted mansion, Tokyo Disney, Hong Kong, magic kingdom, Disney World, Disney Park,   
 
The top 10  bigram words for universal studios are theme park, harry potter, toadstool café, Mario kart, express pas, early access, Studio Hollywood,  super Nintendo, Nintendo World, Universal Studio.   

### 7. Word Embedding Tactic  
 
TF-IDF is a better choice due to its ability to account for the importance of words in a document relative to a collection of documents. It take into account both the term frequency (how often a word appears in a document) and the inverse document frequency (how common or rare a word is across multiple documents). This helps in handling the issue of common words that might occur frequently in many documents but carry little semantic meaning. As both subreddits are related to theme park, the chance of having common words between them would be quite high. Thus TF-IDF is preferred in this scenarios .  
 
In datasets with high dimensionality and sparse features, such as text data, TF-IDF can perform better than CountVectorizer. This is because TF-IDF reduces the impact of common and less informative terms, leading to more effective feature representations. 


### 8. Models Performance


|Model|Train Accuracy Score|Train Accuracy Score|Sensitivity|Specificity|F1|N-Grams|
|---|---|---|---|---|---|---|
|TF-IDF + Support Vector Classification|1.0|0.929648|0.943144|0.916107|0.930693|Bigram|
|TF-IDF + Logr + SVC + NB|1.0|0.929648|0.933110|0.926174|0.930000|Bigram|
|TF-IDF + Multinomial NB|0.999282|0.929648|0.899666|0.959732|0.927586|Bigram|
|TF-IDF + Logistics Regression|1.0|0.929648|0.923077|0.929530|0.926174|Bigram|
|Fast Text|   -  |   -  |0.924623|0.92462|0.924623|Bigram|
|TF-IDF + Random Forest|1.0|0.904523|0.946488|0.862416|0.908507|Bigram|

Based on F1 score, the best model is TF-IDF + Support Vector Classification.

### Fast Text
In the project, I tried using Fast text to train my model. 

FastText is an open-source library developed by Facebook Research for efficient learning of word representations and text classification. It extends the Word2Vec model by incorporating subword information, making it particularly useful for dealing with out-of-vocabulary words. 

While it uses neural network architecture, it is considered a shallow model compared to deep learning architectures like convolutional neural networks (CNNs) or recurrent neural networks (RNNs). However, for this dataset, it did not perform better than most of the models. 

### 9. Model Comparision 

SVC works better in this project because of its property of working well with non-linear relationship data,  effective with high-dimensional data and adaptability to different distributions.  
 
### 9.1 Non - Linear Relationship  
 
The relationship between words and labels can be complex and non-linear, SVMs with appropriate kernels can better model these relationships compared to the linear decision boundaries of logistic regression. 
 
Also, SVC are less sensitive to noise in the data compared to logistic regression. In text classification, where noise can arise from various sources such as misspellings, abbreviations, or inconsistencies in language usage, SVMs can provide better generalization by effectively ignoring noisy data points or outliers.  
 
In text classification, some features (n-grams) may be highly correlated with each other, leading to multicollinearity issues. SVMs are generally less affected by multicollinearity compared to logistic regression, which can lead to more stable and reliable performance. 
 
### 9.2 Effective with High-Dimensional Data 
 
Text classification often involves high-dimensional feature spaces, where each word or n-gram represents a feature. SVMs are known to perform well in high-dimensional spaces  


### 9.3 Adaptability to different distributions 
 
SVMs are non-parametric models and do not make explicit assumptions about the underlying distribution of the data, making them more adaptable to different types of distributions in text data. Naive Bayes, on the other hand, assumes that features follow a specific distribution (e.g., multinomial or Gaussian), which may not always hold true in practice. 


### 10. Improvement 

### 10.1 Scrape more Data to Train and Test The Model 

More data can help in training complex models that have a large number of parameters. Deep learning models, for example, often require large amounts of data to learn intricate patterns effectively.

### 10.2 Remove some highly distinguishable words to generalise the model

Most of the model having a accuracy score of 1. By removing some highly distinguishable words could help to mitigate overfitting and improve the model's ability to generalize to unseen data. By reducing the influence of words that are unique to specific classes or categories, the model can focus on learning more generalizable patterns.

### 10.3 Use cross-validation to evaluate the performance of the model on the training dataset

Cross-validation provides a robust estimate of the model's performance by evaluating it on multiple subsets of the training data. This approach helps in detecting overfitting.

### 12. Recommondation

### 12.1 Marketing Team 

The marketing team can use the top key words above when crafting their marketing materials. Eg “Do and don’t for your first time in Disneyland”, “Is it worth staying at Disney Resort” and etc to engage with their readers. The website content team can also use the top key words to boost the Search Engine Optimization(SEO) score. The higher score, the higher page ranking on the search page leading to higher visibility and increase traffic. 

### 12.2 Trip Curation Team 

The trip curation team can be more detailed when planning, such as what are the must play rides, route to go to play most of the rides, and etc.


### 11. Moving Forward 

By further categorizing these inputs based on sentiment analysis, we can identify patterns and underlying issues affecting our sales performance. For example, if many inquiries express dissatisfaction with the availability of certain services or destinations, we can prioritize addressing these concerns to better meet customer needs and potentially improve sales. Additionally, understanding customer sentiments and preferences can inform strategic decisions, such as adjusting marketing strategies or refining product offerings to better align with market demands. 

### Conclusion 
 
Overall, the project seeks to unlock the value hidden within the company's feedback data, turning it from a passive collection of information into a proactive tool for driving positive change and delivering better outcomes for both the company and its customers. Additionally, understanding customer sentiments and preferences can inform strategic decisions, such as adjusting marketing strategies or refining product offerings to better align with market demands. 

### Citation

Automatic hyperparameter optimization · fasttext. fastText. (n.d.). 
    https://fasttext.cc/docs/en/autotune.html 

Rastogi, K. (2022, November 22). Text cleaning methods in NLP. Analytics Vidhya. 
    https://www.analyticsvidhya.com/blog/2022/01/text-cleaning-methods-in-nlp/ 

>>>>>>> 3b5ffa3cebc8110ef37d3cbbe6d8223e6266aa31
