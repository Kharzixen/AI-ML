# AI-ML

  Artificial Intelligence and Machine learning projects
* ## 1) A* pathfinding algorith,

  ### Project Overview
  Implementation of the A* search algorithm, a versatile and widely used pathfinding algorithm. It's designed to find the shortest path from a starting point to a target point on a graph or grid, taking into account the cost associated with each path. It utilizes a __priority queue__ (heap) to retrieve the most optimal next step and __Euclidean distance__ as the heuristic for estimating the cost to reach the goal. This combination allows for faster pathfinding on grids or graphs.

  ### Algorithm Description
  A* is an informed search algorithm that combines elements of Dijkstra's algorithm and a heuristic approach. It maintains two lists:

  * Open List: A list of nodes to be evaluated.
  * Closed List: A list of nodes that have already been evaluated.

  The algorithm works by exploring the node with the lowest combined cost of the path from the start node to the current node (known as the __"g-cost"__) and the estimated cost from the current node to the goal node       (known as the "__h-cost__" or heuristic cost). The total cost is calculated as __f(n) = g(n) + h(n)__.

  A* continues to expand nodes from the open list until it reaches the target node or exhausts all possible paths. It guarantees finding the shortest path if certain conditions are met:

  * The heuristic function (h(n)) is admissible, meaning it never overestimates the true cost to reach the goal.
  * The graph or grid does not have cycles with negative edge weights.

* ## 2) Bayes Spam filter
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
    <img src="https://i.ibb.co/Jz80kPh/spam2.png" width=30% height=50%>
  </p>

* ## 3) Game with AI using min-max tree
  __...__
  
* ## 4) Optical Character Recognition with preprocessed data
    ###  Project Overview
    Digit recognition is a fundamental problem in the field of optical character recognition and machine learning. This project aims to recognize handwritten digits (0-9) using various machine learning algorithms.

    ### Dataset
    The dataset used in this project consists of 65 long vectors to represent images. Each vector corresponds to a grayscale image of a handwritten digit. The brighter a pixel is, the larger the number in the vector      at that position. The last value (on the 64th index) of the vector is the represented digit.
    __For example:__ The vector bellow represents this handwritten digit: 
      <p float="left">
        <img src="https://i.ibb.co/SRCpLwt/vector.png" width=300px height=200px>
        <img src="https://i.ibb.co/TgSX80Q/number.png" width=300px height=200px>
      </p>
    We have used 3823 different digits for training and 1797 digits for testing the model, these vector are included in the OCR folder.
  
    ### Machine Learning Techniques:
    * __k-Nearest Neighbors (kNN):__ The kNN algorithm is used for both learning and testing on the dataset. It supports two distance metrics: Euclidean distance and Cosine similarity.

    * __Centroid Method:__ This method involves clustering the data by finding centroids for each digit class and then classifying new data points based on their similarity to these centroids.

    * __Gradient Descent (SVM):__ Gradient Descent is used to learn weights and biases for a Support Vector Machine (SVM) classifier. The SVM classifier is used for binary digit classification.
 
    The project evaluates the performance of each algorithm on both training and testing data. It reports classification errors and provides error percentages for each digit class.

