## Introduction
clean label backdoor attack and defense

## Requirements
* Python == 3.7
* `pip install -r requirements.txt`


## Poisoned Sample Generation

If you want to run the Sentence Rewriting model for your data, you need to download the weights.

- Download [Sentence Rewriting model](https://drive.google.com/drive/folders/1jfIgYziCPupYMryO6XjZyiWzOJVDUh7z), and put it into the ``T5`` directory.

```shell
python run_gen_test.py  --do_test
```

Alternatively, you can train your own model.

```shell
python run_gen.py  --do_train --do_eval --do_eval_bleu
```

## Train the Clean Victim Model.

cd to clean_label_textual_backdoor_attack

```shell
python attack/sst_clean.py
```

```shell
python attack/sst_door.py
```

```shell
python attack/sst_attack.py
```

## Train the Clean Defense Model.

cd to clean_label_textual_backdoor_attack_defense

```shell
python attack/sst_t_sne.py
```

The remaining steps are consistent with the Attack model.

## Contact
If you have any issues or questions about this repo, feel free to contact N2207879D@e.ntu.edu.sg.

