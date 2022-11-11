# multiscale-correction


## Step 01: Generate the Adversarial Datasets
```
$ python exp_generate_adversarial_data.py -o outputs/ -s 1234 -a FastGradientMethod
$ python exp_generate_adversarial_data.py -o outputs/ -s 1234 -a ProjectedGradientDescent
$ python exp_generate_adversarial_data.py -o outputs/ -s 1234 -a DeepFool
```