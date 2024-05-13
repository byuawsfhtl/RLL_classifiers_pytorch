import sys
sys.path.append('/home/jroubido/fsl_groups/fslg_census/compute/machine_learning_models/VGG19_classification/branches/Jackson/RLL_VGG19_classifier/new_stuff')

from get_data import get_train_and_test_sets
import pandas as pd
import random

# Generate random field values
fields = [random.choice(['A', 'B', 'C']) for _ in range(10)]

# Creating a DataFrame with random field values
data = {
    'label': fields,
    'path': [f"path{i}" for i in range(1, 11)]
}

df = pd.DataFrame(data)

# labels_and_paths: pd.DataFrame, random_split: bool, stratified_split: bool, test_size: float
list1, list2, list3, list4 = get_train_and_test_sets(df, True, True, .2)
print('yeet')