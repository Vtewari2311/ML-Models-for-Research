# Update Logs

## Change Log (July 26, 2022)


## New Project Structure

```bash
.
├── data/
│    ├── images/
│    │    ├── 
│    │    ├── 
│    │    ├── 
│    │    ├── 
│    │    ├── 
│    ├── dataset.py
├── neural_network/
│    ├── activation/
│    │    ├── __init__.py
│    │    ├── leaky_relu.py
│    │    ├── relu.py
│    │    ├── sigmoid.py
│    │    ├── softmax.py
│    │    ├── tanh.py
│    ├── aux_math/
│    │    ├── __init__.py
│    │    ├── correlate.py
│    │    ├── convolve.py
│    ├── base/
│    │    ├── __init__.py
│    │    ├── activation_mixin.py
│    │    ├── cost_mixin.py
│    │    ├── classifier_mixin.py
│    │    ├── layer_mixin.py
│    │    ├── metadata_mixin.py
│    │    ├── mixin.py
│    │    ├── model_mixin.py
│    │    ├── save_mixin.py
│    │    ├── transform_mixin.py
│    ├── cost/
│    │    ├── __init__.py
│    │    ├── cross_entropy.py
│    │    ├── mean_squared_error.py
│    ├── decomposition/
│    │    ├── __init__.py
│    │    ├── linear_discriminant_analysis.py
│    │    ├── principal_component_analysis.py
│    ├── exceptions/
│    │    ├── __init__.py
│    │    ├── exception_factory.py
│    ├── layers/
│    │    ├── __init__.py
│    │    ├── convolutional.py
│    │    ├── dense.py
│    │    ├── reshape.py
│    ├── metrics/
│    │    ├── __init__.py
│    │    ├── accuracy.py
│    │    ├── accuracy_by_label.py
│    │    ├── average_precision_score.py
│    │    ├── average_recall_score.py
│    │    ├── confusion_matrix.py
│    │    ├── correct_classification_rate.py
│    │    ├── precision_score.py
│    │    ├── recall_score.py
│    ├── model/
│    │    ├── __init__.py
│    │    ├── decision_tree.py
│    │    ├── k_nearest_neighbors.py
│    │    ├── sequential.py
│    ├── model_selection/
│    │    ├── __init__.py
│    │    ├── kfold.py
│    │    ├── repeated_kfold.py
│    │    ├── stratified_kfold.py
│    │    ├── stratified_repeated_kfold.py
│    │    ├── train_test_split.py
│    ├── preprocess/
│    │    ├── __init__.py
│    │    ├── one_hot_encoder.py
│    │    ├── scaler.py
│    │    ├── standardizer.py
│    ├── utils/
│    │    ├── __init__.py
│    │    ├── exceptions_handling.py
│    │    ├── exports.py
│    │    ├── typesafety.py
│    ├── __init__.py
├── __init__.py
├── main.py
├── .gitignore
├── README.md
```
