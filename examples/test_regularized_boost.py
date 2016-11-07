"""An example of testing regularized boosting machines on several  
benchmark datasets. 

   We tested two regularized boosting machines along with the standard 
   one for comparison. Decision trees from sklearn were used as weak 
   classifiers.
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 

from sklearn.tree import DecisionTreeClassifier
from wukong.preprocessing import standardize
from wukong.preprocessing import normalize
from wukong.benchmark import run_benchmark
from wukong.datasets import load_dataset
from wukong.cross_validation import train_test_split
from wukong.ensemble.weight_boosting import AdaBoostClassifier as MyAdaBoostClassifier
from wukong.classifier.naive_bayes import GaussianNB as MyGaussianNB
from wukong.ensemble.regularized_boosting import *


if __name__ == "__main__":
    n_learners = 100
    dtree3 = DecisionTreeClassifier(max_depth=3)
    ab3 = MyAdaBoostClassifier(DecisionTreeClassifier(max_depth=3), 
                               n_learners=n_learners)
    idbc3 = InputDependentBoostClassifier(DecisionTreeClassifier(max_depth=3), 
                                          n_learners=n_learners)
    idbc7 = InputDependentBoostClassifier(DecisionTreeClassifier(max_depth=7), 
                                          n_learners=n_learners)
    smb = SoftMarginBoostClassifier(DecisionTreeClassifier(max_depth=3), 
                                    n_learners=n_learners,
                                    regularization_rate=0.5)


    run_benchmark(learners_obj=[dtree3, ab3, smb, idbc3],
                  learners_name=['CART(3)','AdaBoost(CART(3))','SoftMargin(CART(3))','InputDepBoost(CART(3))'],
                  n_runs=10, 
                  n_folds=10,
                  noise_level=0.0,
                  report_name='SoftMargin_reg_5_runs_10_kfold_10_nlearners_100_noise_00_methods_4',
                  verbose=True)
