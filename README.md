## 1. Reason for augmentation:

 - a. Since the dataset is small, augmentation helps create training samples from limited data.
 - b. By adding augmentation, the model doesn't rely on exact same edge patterns.
 - c. Citation networks have redundancy, hence dropping edges does not change paper topics.
 - d. Edge dropping is the simplest way to start with augmentation.

## 2. Reason for label smoothing:

 - a. To prevent overconfidence & memorization in the training set.
 - b. Our experiments had a big gap between training & test accuracy. Smoothing is a way to keep the model open to learn patterns to reduce the gap between training & test accuracy.

## 3. Focal loss:

 - a. I have not read much about focal loss, but what I found is it helps handle class imbalance.

## 4. Ensemble learning:

 - a. From my previous experience, ensemble learning always gives a better result.
 - b. In this code, I tried to use weighted voting, but I was wondering if we used multiple gnn algorithms, then we could use majority voting.
 - c. I could not run the tests as the code was to heavy on the gpu.

