# multiscale-correction


## Step 00: Generate the Adversarial Datasets
```
$ python 00_generate_adversarial_data.py -o outputs/ -s 1234 -a FastGradientMethod
$ python 00_generate_adversarial_data.py -o outputs/ -s 1234 -a ProjectedGradientDescent
$ python 00_generate_adversarial_data.py -o outputs/ -s 1234 -a DeepFool
```


## Step 01: Generate the Adver
```
$ python 01_generate_models.py -o outputs/ -s 1234
$ python 01_generate_models.py -o outputs/ -s 1435
$ python 01_generate_models.py -o outputs/ -s 8732
$ python 01_generate_models.py -o outputs/ -s 6871
$ python 01_generate_models.py -o outputs/ -s 7823
```



