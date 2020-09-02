from .opt import Adam
from sklearn import metrics
import numpy as np

# ## utility function

# In[45]:


def calculate_f1(y_true, output, thresholds, average):
    """Calculate F1 score.
    Args:
      y_true: (N, (optional)frames_num], classes_num)
      output: (N, (optional)[frames_num], classes_num)
      thresholds: (classes_num,), initial thresholds
      average: 'micro' | 'macro'
    """
    if y_true.ndim == 3:
        (N, T, F) = y_true.shape
        y_true = y_true.reshape((N * T, F))
        output = output.reshape((N * T, F))

    classes_num = y_true.shape[-1]
    binarized_output = np.zeros_like(output)
    #     print('class num:', classes_num)

    for k in range(classes_num):
        binarized_output[:, k] = (np.sign(output[:, k] - thresholds[k]) + 1) // 2

    if average == 'micro':
        return metrics.f1_score(y_true.flatten(), binarized_output.flatten())

    f1_array = []
    for k in range(classes_num):
        f1_array.append(metrics.f1_score(y_true[:, k], binarized_output[:, k]))

    if average == 'macro':
        return np.average(f1_array)
    elif average is None:
        return f1_array
    else:
        raise Exception('Incorrect argument!')

        
def calculate_prec_and_rec(y_true, output, thresholds, average):
    """Calculate F1 score.
    Args:
      y_true: (N, (optional)frames_num], classes_num)
      output: (N, (optional)[frames_num], classes_num)
      thresholds: (classes_num,), initial thresholds
      average: 'micro' | 'macro'
    """
    if y_true.ndim == 3:
        (N, T, F) = y_true.shape
        y_true = y_true.reshape((N * T, F))
        output = output.reshape((N * T, F))

    classes_num = y_true.shape[-1]
    binarized_output = np.zeros_like(output)
    #     print('class num:', classes_num)

    for k in range(classes_num):
        binarized_output[:, k] = (np.sign(output[:, k] - thresholds[k]) + 1) // 2

    if average == 'micro':
        precision = metrics.precision_score(y_true.flatten(), binarized_output.flatten())
        recall = metrics.recall_score(y_true.flatten(), binarized_output.flatten())
        return precision, recall

    elif average is None:
        precision_array, recall_array = [], []
        for k in range(classes_num):
            precision_array.append(metrics.precision_score(y_true[:, k], binarized_output[:, k]))
            recall_array.append(metrics.recall_score(y_true[:, k], binarized_output[:, k]))
        return precision_array, recall_array
    else:
        raise Exception('Incorrect argument!')

        

def calculate_at_gradient(y_true, output, thresholds, average):
    """Calculate gradient of thresholds numerically.
    Args:
      y_true: (N, (optional)frames_num], classes_num)
      output: (N, (optional)[frames_num], classes_num)
      thresholds: (classes_num,), initial thresholds
      average: 'micro' | 'macro'
    Returns:
      grads: vector
    """
    f1 = calculate_f1(y_true, output, thresholds, average)
    classes_num = len(thresholds)

    delta = 0.01
    grads = []
    #     print("calculate_at_gradient, f1:", f1)

    for k, threshold in enumerate(thresholds):
        new_thresholds = thresholds.copy()
        cnt = 0
        while cnt < 10:
            cnt += 1
            new_thresholds[k] += delta
            f1_new = calculate_f1(y_true, output, new_thresholds, average)
            if f1_new != f1:
                break

        grad = (f1_new - f1) / (delta * cnt)
        grads.append(grad)

    return grads


def calculate_prec_and_rec_gradient(y_true, output, thresholds, average):
    """Calculate gradient of thresholds numerically.
    Args:
      y_true: (N, (optional)frames_num], classes_num)
      output: (N, (optional)[frames_num], classes_num)
      thresholds: (classes_num,), initial thresholds
      average: 'micro' | 'macro'
    Returns:
      grads: vector
    """
#     f1 = calculate_f1(y_true, output, thresholds, average)
    prec, rec = calculate_prec_and_rec(y_true, output, thresholds, average)

    classes_num = len(thresholds)

    delta = 0.01
    grads = []
    #     print("calculate_at_gradient, f1:", f1)

    for k, threshold in enumerate(thresholds):
        new_thresholds = thresholds.copy()
        cnt = 0
        while cnt < 10:
            cnt += 1
            new_thresholds[k] += delta
#             f1_new = calculate_f1(y_true, output, new_thresholds, average)
            prec_new, rec_new = calculate_prec_and_rec(y_true, output, new_thresholds, average)
#             if f1_new != f1:
#                 break
            if prec_new != prec or rec_new != rec:
                break

        grad = 0.5 * (prec_new - prec + rec_new - rec) / (delta * cnt)
#         grad = (0.2*(prec_new - prec) + 0.8*(rec_new - rec)) / (delta * cnt)
        grads.append(grad)

    return grads


def optimize_at_with_gd(y_true, output, thresholds, average):
    """Optimize thresholds for AT.
    Args:
      y_true: (N, (optional)frames_num], classes_num)
      output: (N, (optional)[frames_num], classes_num)
      thresholds: (classes_num,), initial thresholds
      average: 'micro' | 'macro'
    Returns:
      metric: float
      thresholds: vector
    """
    opt = Adam()
    opt.alpha = 1e-2
    metric = calculate_f1(y_true, output, thresholds, average)
    metric_asfo_epoch = [metric]
    for i in range(100):
        grads = calculate_at_gradient(y_true, output, thresholds, average)
        #         if i==0: print("grads:", grads)
        grads = [-e for e in grads]
        thresholds = opt.GetNewParams(thresholds, grads)
        metric = calculate_f1(y_true, output, thresholds, average)
        if i % 50 == 0:
            print('Iteration: {}, Score: {:.3f}, thresholds: {}'.format(
                i, metric, np.array(thresholds)))
        metric_asfo_epoch.append(metric)

    return metric, thresholds, metric_asfo_epoch


def optimize_prec_and_rec_with_gd(y_true, output, thresholds, average):
    """Optimize thresholds for AT: 0.5(precision+recall)
    Args:
      y_true: (N, (optional)frames_num], classes_num)
      output: (N, (optional)[frames_num], classes_num)
      thresholds: (classes_num,), initial thresholds
      average: 'micro' | 'macro'
    Returns:
      metric: float
      thresholds: vector
    """
    opt = Adam()
    opt.alpha = 1e-2
    metric = calculate_f1(y_true, output, thresholds, average)
    metric_asfo_epoch = [metric]
    for i in range(100):
        grads = calculate_prec_and_rec_gradient(y_true, output, thresholds, average)
        #         if i==0: print("grads:", grads)
        grads = [-e for e in grads]
        thresholds = opt.GetNewParams(thresholds, grads)
        metric = calculate_f1(y_true, output, thresholds, average)
        if i % 50 == 0:
            print('Iteration: {}, Score: {:.3f}, thresholds: {}'.format(
                i, metric, np.array(thresholds)))
        metric_asfo_epoch.append(metric)

    return metric, thresholds, metric_asfo_epoch

