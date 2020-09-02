def heu_threshold_opti(y_true, y_pred, init_thresholds):
    """optimization routine to optimize AT F-score

    We start from default 0.5 thresholds and search for better ones for each of the ten classes"""

    bin_y_pred = binarize_probs(y_pred, init_thresholds)
    #     bin_y_pred = y_pred.copy()
    #     bin_y_pred[bin_y_pred > 0.5] = 1
    #     bin_y_pred[bin_y_pred <= 0.5] = 0

    sk_f1 = np.asarray(f1_score(y_true, bin_y_pred, average=None))
    #     sk_f1 = calculate_f1(y_true, y_pred, thresholds=init_thresholds, average=None)

    macro_iteration = 80
    micro_interation = 400

    history = {
        "best_f1": [[] for _ in range(10)],
        "f1": []
    }

    #     best_thresholds = [0.5] * 10
    best_thresholds = init_thresholds
    best_f1 = sk_f1.copy()

    #     progress = tqdm.tqdm_notebook(total = macro_iteration * micro_interation)
    for M in range(macro_iteration):
        #         thresholds = [0.5] * 10
        thresholds = init_thresholds
        delta_ratio = 0.2
        delta_decay = 1e-4

        for m in range(micro_interation):
            bin_y_pred = y_pred.copy()

            # calc new threhsold
            r = np.array([np.random.normal(t, 0.6) for t in thresholds])
            delta = r * delta_ratio
            new_thresholds = thresholds + delta
            delta_ratio -= delta_decay

            # apply threshold
            bin_y_pred = binarize_probs(y_pred, new_thresholds)
            #             bin_y_pred[bin_y_pred > new_thresholds] = 1
            #             bin_y_pred[bin_y_pred <= new_thresholds] = 0

            # calc new f1
            new_f1 = f1_score(y_true, bin_y_pred, average=None)
            #             new_f1 = calculate_f1(y_true, y_pred, thresholds=new_thresholds, average=None)
            history["f1"].append(new_f1)

            # check
            for i in range(10):
                if new_f1[i] > best_f1[i]:
                    #                     print(new_f1[i], " > ", best_f1[i])
                    best_f1[i] = new_f1[i]
                    best_thresholds[i] = new_thresholds[i]
                    thresholds[i] = best_thresholds[i]
                    history["best_f1"][i].append(best_f1[i])

                    #             progress.update()
    return sk_f1, best_f1, best_thresholds, history