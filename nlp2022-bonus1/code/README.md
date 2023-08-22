# NLP 2022 - Bonus Exercise 1

Before reading this readme please read the provided slides `NLP_2022-Bonus_Exercise_1.pdf`.

## Structure

Folder structure:

```
- data/
    - train.jsonl
    - dev.jsonl
    - test.jsonl
- gold/
    - gold_dev.tsv
- predictions/
    - random_baseline_dev.tsv
- scorer.py
- NLP_2022-Bonus_Exercise_1.pdf
- README.md
```

## Objective

You have to predict the labels for the `data/test.jsonl` dataset 
and produce a file in a `tsv` format named `predictions_test.tsv`. 

example:

```
SAMPLE_NUMERIC_ID{TAB}predicted_label
SAMPLE_NUMERIC_ID{TAB}predicted_label
```
replace {TAB} with `\t`

real example:

```
83565	sci/tech
193950	business
79069	media
79229	environment
```

You can find an example of the file format `predictions/random_baseline_dev.tsv`

### What to deliver:
You have to submit a `zip` file named `{MATRICOLA}_{SURNAME}_bonus-nlp-2022.zip` 
containing the `predictions` and `code` folders.

File name example:

```
1381242_rossi_bonus-nlp-2022.zip
```

ZIP structure:

```
- predictions/
    - predictions_test.tsv
- code/
    - YOUR CODE FILES
```

## Dataset format

The dataset is located in `./data` folder, and it is in a `jsonl` file format.
Each line contains a json sample in the following structure:

```json lines
{
   "text":"TEXT OF THE SAMPLE",
   "label":"LABEL OF THE SAMPLE",
   "id": SAMPLE_NUMERIC_ID
}
```

Real sample:

```json lines
{
   "text":"Chargers to start Brees in Houston, beyond com. The San Diego Chargers announced on Monday that Drew Brees will start the 2004 opener against the Houston Texans at Reliant Stadium.",
   "label":"sports",
   "id":194207
}
```

The test dataset **DOES NOT** contain the `label` field. 

## Evaluate your answers

To evaluate your answers we will use the `scorer.py`. We **warmly** invite 
you to verify that your predictions are formatted correctly using the dev dataset.

Usage example:

```
python3 scorer.py [-h] --prediction_file PREDICTION_FILE --gold_file GOLD_FILE
```

Real example on random predictions:

```
python3 scorer.py --prediction_file predictions/random_baseline_dev.tsv --gold_file gold/gold_dev.tsv
```

output:

```json lines
> {'err_rate': '93.02'}
```

We expect that your predictions produce an error rate **lower** than the random baseline.



