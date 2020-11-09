# SGLThresh

## Surrogate Gradient Learning for estimating multilabel classification thresholds

In this work, we propose to optimize thresholds used to do multilabel classification. We propose to use gradient descent to optimize F-score (aka F1) by estimating the best threshold values using gradient descent. 

Since F1 is not differentiable due to the thresholding operation, we propose to use a gradient proxy, hence the name Surrogate Gradient Learning thresholds, in short SGLThresh.

In this repo, you will find three notebooks to do so on three multilabel audio event detection: DCASE 2017, DCASE 2019 and AudioSet. We use as input to our method predictions made by state-of-the-art neural networks. We optimize the thresholds on a validation subset and then apply them to an evaluation subset:

![datasets](/datasets.png)

In each of these notebooks:

* sgl_dcase2017.ipynb
* sgl_dcase2019.ipynb
* sgl_audioset.ipynb

there is our code to run SGLThresh and to compare to other methods: static thresholds, numThresh (estimation with numerical gradients) and heuThresh (heuristic-based optimization).

We used torch 1.4 and numpy 1.18 to run the experiments. We procide a YAML conda env file, to install the environment please do:

    conda env create -f environment.yml

SGLThresh outperforms the other methods and  our PyTorch implementation is very fast:

![results](/results.png)

