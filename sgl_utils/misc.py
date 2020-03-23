
def predict(mydataloader, y, model):
    model.eval()

    all_outputs = torch.zeros_like(y)

    for i_batch, sample_batched in enumerate(mydataloader):

        X_batch, y_batch = sample_batched
        outputs = model(X_batch)
        B = X_batch.size()[0]
        all_outputs[ i_batch *B:( i_batch +1 ) *B ] =outputs
    return all_outputs


def print_thresholds(thresh, nb_classes):
    s = ''
    for c in range(nb_classes): s+='%.4f '% thresh[ c ]
    print("auto_thresholds", s)


def print_thresholds_fh(thresh, nb_classes, fh):
    s = ''
    for c in range(nb_classes):
        s+= '%.4f '% thresh[ c ]
    fh.write("auto_thresholds: %s\n"%s)


def compute_instance_F1(gt_y, preds):
    num = gt_y*preds
    num = 2*np.sum(num, axis=1)
    den = np.sum(gt_y, axis=1) + np.sum(preds, axis=1)
    return np.mean(num/den)


def unit_test(gt_y, preds):
    print("%.2f"% compute_instance_F1 (gt_y, preds))


def print_scores(gt_y, preds):
    print(classification_report(gt_y, preds, digits=3))
    set_accuracy = accuracy_score(gt_y, preds)
    print('set acc: %.3f'%(set_accuracy))


def print_scores_fh(gt_y, preds, fh):
    fh.write(classification_report(gt_y, preds, digits=3))
    set_accuracy = accuracy_score(gt_y, preds)
    fh.write('\nset acc: %.3f\n'%(set_accuracy))


# def print_scores(gt_y, preds):
#     p2, r2, fscore2, support = precision_recall_fscore_support(gt_y, preds, pos_label=1, average='micro')
#     p3, r3, fscore3, support = precision_recall_fscore_support(gt_y, preds, pos_label=1, average='macro')
#     set_accuracy = accuracy_score(gt_y, preds)
#     print('micro: p:%.2f r:%.2f f1:%.2f'%(100.*p2, 100.*r2, 100.*fscore2))
#     print('macro: p:%.2f r:%.2f f1:%.2f'%(100.*p3, 100.*r3, 100.*fscore3))
#     print('set acc:%.2f instance-F1:%.2f'%(100.*set_accuracy, 100*compute_instance_F1(gt_y, preds)))

def compute_accuracy_from_numpy_tensors(gt_y_numpy, preds_numpy):
    acc_per_class = sum(gt_y_numpy==preds_numpy)/len( gt_y_numpy)
    gt_y_numpy_vec = np.reshape(gt_y_numpy, -1)
    preds_numpy_vec = np.reshape(preds_numpy, -1)
    acc = sum(gt_y_numpy_vec==preds_numpy_vec)/len( gt_y_numpy_vec)
    print( "Acc per class:", acc_per_class)
    print("Acc: %.3f"%acc)

# dummy_y = np.array([[1,0,0], [0,1,0]])
# dummy_preds = np.array([[1,0,0], [0,0,0]])
# unit_test(dummy_y, dummy_preds)
