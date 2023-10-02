# AI-ML

Artificial Intelligence and Machine learning projects, created during my university course
# 1) A* pathfinding algorith,
__...__
# 2) Bayes Spam filter
  ### Project Overview
  The Bayesian Spam Filter project aims to identify whether an email is spam or not using Bayesian probability theory. It combines both supervised and unsupervised training methods to improve accuracy in classifying emails. The project utilizes a dataset of labeled emails to train a supervised classifier and employs unsupervised techniques like Naive Bayes to enhance spam detection.
  ### Features
  The project uses a labeled dataset of emails, which can be found in the `data` directory. Make sure to update the dataset with your own email data or obtain a suitable dataset for further training and testing.
  ### Supervised and Unsupervised Training
  The supervised classifier is trained using labeled data to learn patterns and features that distinguish spam from non-spam emails. It is essential to have a well-labeled dataset for effective supervised training.
  
  The unsupervised Bayesian filter leverages Bayesian probability and Naive Bayes techniques to classify emails without relying on labeled data. It calculates the likelihood of an email being spam or ham based on learned probabilities from the training data.
  
  ### Additive/Laplace/Lidstone smoothing
  Additive, Laplace, or Lidstone smoothing is a technique used to address the problem of zero probabilities when applying Bayesian probability-based algorithms like Naive Bayes, especially in situations with limited data. In the context of the Bayesian Spam Filter, smoothing can help improve the robustness of the classifier.
  
  In the Bayesian Spam Filter, smoothing is applied to calculate the probabilities of tokens in both spam and non-spam emails. When a feature is absent in one class but present in another, it can lead to a probability of zero, causing issues during classification.
  
  Additive/Laplace/Lidstone smoothing addresses this by adding a small constant, **alpha** to the counts of each feature for both classes. This ensures that no probability becomes zero, and it prevents the model from becoming overly confident about specific features.
  
  ### Metrics
  Evaluate the performance of the Bayesian Spam Filter using metrics like accuracy, precision, False positive/False negative ratio. 
  
  <p float="left">
    <img src="https://i.ibb.co/2qHK9NF/spam1.png" width=30% height=50%>
    <img src="https://i.ibb.co/Jz80kPh/spam2.png" width=30% height=30%>
  </p>

# 3) Game with AI using min-max tree
__...__
# 4) Optical Character Recognition with preprocessed data
__...__

