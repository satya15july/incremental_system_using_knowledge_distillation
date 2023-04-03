# Building Incremental System Using Knowledge Distillation

## Introduction

![il_approch_FC drawio](https://user-images.githubusercontent.com/22910010/229521934-9ebd4022-bacb-4614-81d8-5c3599dc69cf.png)

The major problem with AI model is how to preserve the learn parameters when more visula information is incorporated in to the AI model from time to time.

Knowledge distillation is one of the technique majorly used for developing incremental learning system.

![IL_KD](https://user-images.githubusercontent.com/22910010/229520825-7bc62e2b-a6b0-42f8-9894-5e74698efe7b.png)

The is how Grad-Cam(Gradient-Weighted Class Activation Map) can be used for preserving knowledge between Teacher Network and Student Network.

![IL_main drawio(3)](https://user-images.githubusercontent.com/22910010/229521347-336a9252-8b46-4752-87c6-79bfbb2fd07d.png)

Please refer my Medium blog for more information.

## Dependencies
- Pytorch
- Python3

## How to Use

### Build Teacher Model

`python3 train_teacher_model.py`

#### Build Incremental System by Using Student Model:

##### Create Student Model:

`python3 incremental_learning_system.py --teacher <teacher model path>`

Use Resume option to train more.You need to change the "RESUME" flag to True before running the below command.

`python3 incremental_learning_system.py --teacher <teacher model path> --student <student model>`


### Run the Demo

`python3 demo.py --teacher <teacher model path> --student <student model>`

![IL_data drawio](https://user-images.githubusercontent.com/22910010/229521130-be2cf7e4-998d-466e-bf70-b919448d66e6.png)

### Example

Let's take one example.
Using 2 classes to verify the model is faster and easier.So let's take the example of 2 Old classes + 2 New Classes.

Create Teacher(with 2 classes) and Student model,with 2(old) + 2(new) classes.But train the Student model with only new classes.

#### Train Teacher model

- Use the class `select_classes_2_1 = ['apple', 'aquarium_fish']` from config.py.
- Use the class name in train_teacher_model.py

    `class_indices = np.isin(cifar_dataset.targets, [cifar_dataset.class_to_idx[c] for c in config.select_classes_2_1])`

- Train the Teacher Model
    `python3 train_teacher_model.py`

#### Train Student Model with 2 new Classes

- Use the class `select_classes_2_2 = ['baby', 'bear']` from config.py.

- Use the new class name in incremental_learning_system.py.
    `class_indices = np.isin(cifar_dataset.targets, [cifar_dataset.class_to_idx[c] for c in config.select_classes_2_2])`

    ` 
    BATCH_SIZE = 128

    EPOCH = 100

    SAVE_INTERVAL = 2000

    RESUME = False

    OLD_CLASSES = 2 # Make sure you change this to 2

    NEW_CLASSES = 2 # Make sure you change this to 2

    TOTAL_CLASSES = OLD_CLASSES + NEW_CLASSES

    `

- Train the Student Model with 2 new classes
    `python3 incremental_learning_system.py --teacher <teacher model path>`

   Check the settings before training present in config.py

    `
    BATCH_SIZE = 128

    EPOCH = 5000

    SAVE_INTERVAL = 2000

    RESUME = False

    OLD_CLASSES = 2 # Make sure you change this to 2

    NEW_CLASSES = 2 # Make sure you change this to 2

    TOTAL_CLASSES = OLD_CLASSES + NEW_CLASSES

    `

    If you want to resume the training,then change the RESUME flag to True in config.py and then execute

    `python3 incremental_learning_system.py --teacher <teacher model path> --student <student model>`

- Run Demo

    Use Old classes in demo.py to verify how Student model is performing with Old classes.This is our objective,Am i right.

    __(Note: We have not trained Student model with Old Classes)__

    So do these changes in demo.py

    - Use `select_classes_2_1 = ['apple', 'aquarium_fish']` in demo.py.

        `class_indices = np.isin(cifar_dataset.targets, [cifar_dataset.class_to_idx[c] for c in config.select_classes_2_1])`
    - Run
        `python3 demo.py --teacher <teacher model path> --student <student model>`


You can try ResNet-110(which is supported in my repo) instead of ResNet-18 and can try 20 classes initially,then add more classes incrementally in to your model.

Then you can use this technique in some real time project.

## Conclusion

Here i demonstrated how Knowledge Distillation can be used to develop incremental system for classification problem.
This technique can be extended to other problems such as segmentation, object detection etc.


## References
- https://arxiv.org/abs/1811.08051.
- https://arxiv.org/abs/1610.02391.
- https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/.
- https://machinelearningmastery.com/update-neural-network-models-with-more-data/.

---
Reach me @

[LinkedIn](https://www.linkedin.com/in/satya1507/) [GitHub](https://github.com/satya15july) [Medium](https://medium.com/@satya15july_11937)

