# Reduce function
from functools import reduce



'''
Evaluation Metric: Quadratic Weighted Kappa
'''
'''
Confusion Matrix used to help compute the Quadratic Weighted Kappa
'''
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """

    # Assert that both samples must be of equal length
    assert(len(rater_a)==len(rater_b))

    # If min_rating is not specified, get the minimum rating from dataset
    if min_rating is None:
        min_rating = min(reduce(min, rater_a), reduce(min, rater_b))

    # If max_rating is not specified, get the maximum rating from dataset
    if max_rating is None:
        max_rating = max(reduce(max, rater_a), reduce(max, rater_b))

    # Number of possible ratings    
    num_ratings = max_rating - min_rating + 1

    # Initialise the confusion matrix to all 0
    conf_mat = [[0 for i in range(num_ratings)] for j in range(num_ratings)]

    # Append at the value in confusion matrix 
    for a,b in zip(rater_a,rater_b):
        conf_mat[a-min_rating][b-min_rating] += 1
    
    return conf_mat



'''
Histogram to help compute the Quadratic Weighted Kappa
'''
def histogram(ratings, min_rating=None, max_rating=None):
    '''
    Returns the counts of each type of rating that a rater made
    '''

    # If min_rating is not specified, get the minimum rating from dataset
    if min_rating is None: min_rating = reduce(min, ratings)

    # If max_rating is not specified, get the maximum rating from dataset
    if max_rating is None: max_rating = reduce(max, ratings)
    
    # Number of possible ratings
    num_ratings = max_rating - min_rating + 1
    
    # Initialise all possible ratings to 0
    hist_ratings = [0 for x in range(num_ratings)]
    
    # Append count at rating - min_rating to hist_ratings
    for r in ratings:
        hist_ratings[r-min_rating] += 1
    
    return hist_ratings



'''
The Main Function that computes the Evaluation Metric: Quadratic Weighted Kappa
'''
def quadratic_weighted_kappa(rater_a, rater_b, min_rating = None, max_rating = None):
    '''
    Calculates the quadratic weighted kappa value, which is a measure of 
    inter-rater agreement between two rater that provide discrete numeric
    ratings.  Potential values range from -1 (representing complete 
    disagreement) to 1 (representing complete agreement).  A kappa value 
    of 0 is expected if all agreement is due to chance.
    
    scoreQuadraticWeightedKappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
   
    score_quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    '''

    # Assert that both samples must be of equal length
    assert(len(rater_a) == len(rater_b))

    # If min_rating is not specified, get the minimum rating from dataset
    if min_rating is None:
        min_rating = min(reduce(min, rater_a), reduce(min, rater_b))

    # If max_rating is not specified, get the maximum rating from dataset
    if max_rating is None:
        max_rating = max(reduce(max, rater_a), reduce(max, rater_b))

    # Get confusion matrix using helper function
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)

    # Get number of possible ratings from the confusion matrix
    num_ratings = len(conf_mat)

    # Get the number of ratings
    num_scored_items = float(len(rater_a))

    # Histogram for Rater A
    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    
    # Histogram for Rater B
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    # Initialise numerator and denominator
    numerator = 0.0
    denominator = 0.0

    # In the range of all possible ratings
    for i in range(num_ratings):
        # Traverse each row of the Confusion Matrix
        for j in range(num_ratings):

            # Get the expected score by using: ((rater A) * (rater B)) / Number of samples
            expected_count = (hist_rater_a[i]*hist_rater_b[j] / num_scored_items)
            
            # d = (#row - #column)^2 / #possible_ratings^2
            d = pow(i-j,2.0) / pow(num_ratings-1, 2.0)

            # Numerator = Numerator + (d * confu_matrix / #rating_samples)
            numerator += d*conf_mat[i][j] / num_scored_items
            
            # Denominator = Denominator + (d * expected_count / #rating_samples)
            denominator += d*expected_count / num_scored_items

    # Return inverse of the calculated Numerator/Denominator
    return 1.0 - numerator / denominator