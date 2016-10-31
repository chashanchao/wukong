"""Regularized boosting  

"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 


import numbers
import numpy as np
import copy 
from ..base import clone
from ..ensemble.weight_boosting import BaseWeightBoosting


__all__=[
    "SoftMarginBoostClassifier","InputDependentBoostClassifier", "ARBoostClassifier"
]


EPSILON = 1e-10

class SoftMarginBoostClassifier(BaseWeightBoosting):
    """Soft margin boosting described in Ratsch's paper "Soft margins for adaboost" 
            
    Parameters:
    -----------
    base_learner.  object 
        At each iteration, a base learner is trained on the weighted samples. In the end, 
        the weighted combination of all the base learner forms the final strong classifier.              
            
    n_learners. int, optional (default=100)
        Number of the base learners.
    
    Reference:
    ---------
    [1]G. RATSCH, T. ONODA , AND K. R. MULLER, Soft margins for adaboost, Machine Learning, 42 
    (2001), pp. 287-320. 
    """
    def __init__(self,
                 base_learner=None,
                 learner_params=tuple(), 
                 n_learners=100,
                 learning_rate=1.0,
                 regularization_rate=0.5,
                 verbose=False):        
        super(SoftMarginBoostClassifier, self).__init__(base_learner, 
                                                        learner_params, 
                                                        n_learners, 
                                                        verbose)        
       
        self.learning_rate = learning_rate 
        self.weighted_errors = []
        
        # 
        # Response of all base learners on the samples. 
        #         x0,      x1, ... ,  xn-1 
        #  h_0    
        #  h_1
        #  ...
        #  h_T-1 
        #    
        self.base_learners_response = []        
        
        # 
        # Response of all base learners on the samples. 
        #         x0,      x1, ... ,  xn-1 
        #  f_0    
        #  f_1
        #  ...
        #  f_T-1 
        #    
        self.staged_strong_learner_response = []

        # 
        # Sample weights for all the iterations 
        #         x0,   x1, ... ,  xn-1 
        #  f_0   
        #  f_1
        #  ...
        #  f_T-1 
        # 
        self.staged_sample_weights = []
        
        #
        # 0.5 is used in the mentioned paper 
        #
        self.regularization_rate = regularization_rate
     

    def _check_stop_criteria(self):
        """Overridden. Check all the stop criteria, return True if anyone of them is satified, 
        otherwise return False"""
        early_stopping = False
        # Return true, if no improvement any more. 
        error = np.abs(self.h_y - self.y) * 0.5   # 1, if incorrect. 0, if correct.
        if np.sum(error) < EPSILON:
            early_stopping = True
        
        return early_stopping or super(SoftMarginBoostClassifier, self)._check_stop_criteria()


    def _update_sample_weights(self):
        """Overridden. Update samples' weight."""
        #
        # Compute weighted error of the latest base learner
        #               
        error = np.abs(self.h_y - self.y) * 0.5   # 1, if incorrect. 0, if correct.
        weighted_error = np.inner(error, self.sample_weights)

        #
        # Compute base learners' weights
        #               
        learner_weight = 0.5 * np.log(
            (1.0 - weighted_error + EPSILON) / (weighted_error + EPSILON)
        )        

        #
        # Compute the regularization term: influence of a pattern
        #
        if self.staged_sample_weights == []:
            sample_influence = 0.0
        else:
            ws = np.array(self.staged_sample_weights)
            ats = np.reshape(np.array(self.learner_weights_), (len(self.learner_weights_), 1))        
            sample_influence = np.sum(ws*ats, axis=0)
          
        # 
        # Compute sample weights for next iteration
        #
        self.sample_weights *= np.exp(
            learner_weight * (error - 0.5) * 2.0 * self.learning_rate -
                self.regularization_rate*sample_influence                                      
        )
        self.sample_weights /= np.sum(self.sample_weights)
                    
        #
        # Store the changes
        #
        self.learner_weights_.append(learner_weight)
        self.weighted_errors.append(weighted_error)

        self.base_learners_response.append(np.copy(self.h_y))
        self.staged_sample_weights.append(np.copy(self.sample_weights))
                 
        #
        # Compute response of the strong learner so far 
        #       
        #if self.staged_strong_learner_response == []:
        #    strong_learner_response = learner_weight * self.h_y
        #else:
        #    strong_learner_response = self.staged_strong_learner_response[-1] + \
        #                              learner_weight * self.h_y

        # 
        # Compute sample weights for next iteration
        #
        #f_y = strong_learner_response
        #W = np.exp(-f_y*self.y - self.regularization_rate*sample_influence)
        #self.sample_weights = W/np.sum(W)        
        
        #
        # Store the changes
        #
        #self.learner_weights_.append(learner_weight)
        #self.weighted_errors.append(weighted_error)
        #self.base_learners_response.append(np.copy(self.h_y))
        #self.staged_sample_weights.append(np.copy(self.sample_weights))
        #self.staged_strong_learner_response.append(strong_learner_response)



class InputDependentBoostClassifier(BaseWeightBoosting):
    """Adaboost for classification.
            
    Parameters:
    -----------
    base_learner.  object 
        At each iteration, a base learner is trained on the weighted samples. In the end, 
        the weighted combination of all the base learner forms the final strong classifier.              
            
    n_learners. int, optional (default=100)
        Number of the base learners.
    
    Reference:
    ---------
    [1] R. JIN, Y. LIU, L. SI, J. CARBONELL, AND A. G. HAUPTMANN, A new boosting algorithm using
        input-dependent regularizer, in Proceedings of Twentieth International Conference on Machine
        Learning(ICML03), AAAI Press, 2003.
    """
    def __init__(self,
                 base_learner=None,
                 learner_params=tuple(), 
                 n_learners=100,
                 learning_rate=1.0,
                 input_dependent_factor=0.5,
                 verbose=False):        
        super(InputDependentBoostClassifier, self).__init__(base_learner, 
                                                            learner_params, 
                                                            n_learners, 
                                                            verbose)        
       
        self.learning_rate = learning_rate 
        self.weighted_errors = []
        
        # 
        # Response of all base learners on the samples. 
        #         x0,      x1, ... ,  xn-1 
        #  h_0    
        #  h_1
        #  ...
        #  h_T-1 
        #    
        self.base_learners_response = []        
        
        # 
        # Response of all base learners on the samples. 
        #         x0,      x1, ... ,  xn-1 
        #  f_0    
        #  f_1
        #  ...
        #  f_T-1 
        #    
        self.staged_strong_learner_response = []

        # 
        # Sample weights for all the iterations 
        #         x0,   x1, ... ,  xn-1 
        #  f_0   
        #  f_1
        #  ...
        #  f_T-1 
        # 
        self.staged_sample_weights = []
        
        #
        # 0.5 is used in the mentioned paper 
        #
        self.beta = input_dependent_factor  
        
        self.input_dependent_factor = []


    def _check_stop_criteria(self):
        """Overridden. Check all the stop criteria, return True if anyone of them is satified, 
        otherwise return False"""
        early_stopping = False
        # Return true, if no improvement any more. 
        error = np.abs(self.h_y - self.y) * 0.5   # 1, if incorrect. 0, if correct.
        if np.sum(error) < EPSILON:
            early_stopping = True
        
        return early_stopping or super(InputDependentBoostClassifier, self)._check_stop_criteria()


    def _update_sample_weights(self):
        """Overridden. Update samples' weight."""
        #
        # Compute weighted error of the latest base learner
        #               
        error = np.abs(self.h_y - self.y) * 0.5   # 1, if incorrect. 0, if correct.
        weighted_error = np.inner(error, self.sample_weights)

        #
        # Compute base learners' weights
        #               
        learner_weight = 0.5 * np.log(
            (1.0 - weighted_error + EPSILON) / (weighted_error + EPSILON)
        )        
                
        #
        # Compute response of the strong learner so far 
        #       
        if self.staged_strong_learner_response == []:
            strong_learner_response = learner_weight * self.h_y
        else:
            strong_learner_response = self.staged_strong_learner_response[-1] + \
                                      learner_weight * self.h_y

        #
        # Compute the normalization factor of input dependent items
        #
        idnorm_factor=10*np.mean(np.exp(-np.abs(self.beta*strong_learner_response))) 
        
        # 
        # Compute sample weights for next iteration
        #
        f_y = strong_learner_response
        #W = np.exp(-f_y*self.y)
        W = np.exp(-f_y*self.y - np.abs(self.beta*f_y))/idnorm_factor
        self.sample_weights = W/np.sum(W)        
        
        #
        # Store the changes
        #
        self.learner_weights_.append(learner_weight)
        self.weighted_errors.append(weighted_error)
        self.base_learners_response.append(np.copy(self.h_y))
        self.staged_sample_weights.append(np.copy(self.sample_weights))
        self.staged_strong_learner_response.append(strong_learner_response)
        self.input_dependent_factor.append(idnorm_factor)


    def predict(self, X):
        """Overridden. Predict y given a set of samples X.   
            
        Parameters:
        -----------
        X. array-like. shape [n_samples, n_features]
            
        Returns:
        --------
        y. array-like shape [n_samples]                
        """
        f = 0 
        for k in range(len(self.base_learners_)):
            h_y = self.base_learners_[k].predict(X)
            f += np.exp(-np.abs(self.beta*f))*h_y*self.learner_weights_[k]
            
        return np.sign(f)    
    

class ARBoostClassifier(BaseWeightBoosting):
    """AR-Boost for classification.
            
    Parameters:
    -----------
    base_learner.  object 
        At each iteration, a base learner is trained on the weighted samples. In the end, 
        the weighted combination of all the base learner forms the final strong classifier.              
            
    n_learners. int, optional (default=100)
        Number of the base learners.
    
    Reference:
    ---------
    Baidya Nath Saha etc. AR-Boost: Reducing Overfitting by a Robust Data-Driven Regularization 
    Strategy. European Conference, ECML PKDD 2013, Prague, Czech Republic, September 23-27, 2013, 
    Proceedings.
    """
    def __init__(self,
                 base_learner=None,
                 learner_params=tuple(), 
                 n_learners=100,
                 relax_rate=4.0):
        BaseWeightBoosting.__init__(self,
                                    base_learner, 
                                    learner_params,
                                    n_learners)
        
        if not isinstance(relax_rate,(numbers.Number, np.number)) or relax_rate < 1.0:
            raise ValueError(
                "relax_rate must be a number and bigger than 1.0"
                )
            
        
        self.alpha = 0              
        self.predict_error = []        
        self.weighted_errors = []
        
        self.maha_distance = []
        self.all_predict_ys = []
        self.all_sample_weights = []
        


    def _check_stop_criteria(self):
        """Check all the stop criteria, return True if anyone of them is satified, 
        otherwise return False"""
        early_stopping = False
        # Return true, if no improvement any more. 
        error = np.abs(self.h_y - self.y) * 0.5   # 1, if incorrect. 0, if correct.
        if np.sum(error) < EPSILON:
            early_stopping = True
        
        return early_stopping or BaseWeightBoosting._check_stop_criteria(self)


    def _update_sample_weights(self):
        """Override method. Update samples' weight."""               
        error = np.abs(self.h_y - self.y) * 0.5   # 1, if incorrect. 0, if correct.
        weighted_error = np.inner(error, self.sample_weights)
        rho = 4.0 #np.exp(0.5*0.5) # key param of AR-Boost. 4.0 is given in their paper.
        if weighted_error >= rho/(rho+1):
            self.early_stopping_ = True
            return
        learner_weight = 0.5 * np.log(
                            rho * (1.0 - weighted_error + EPSILON) / (weighted_error + EPSILON)
                        )        

        self.sample_weights *= np.exp(2.0*learner_weight*error)
        self.sample_weights /= np.sum(self.sample_weights)
                    
        self.learner_weights_.append(learner_weight)
        self.weighted_errors.append(weighted_error)

        self.all_predict_ys.append(np.copy(self.h_y))
        self.all_sample_weights.append(np.copy(self.sample_weights))
  