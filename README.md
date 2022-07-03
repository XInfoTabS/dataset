# Evaluating Multilingual Tabular Natural Language Inference
The Official dataset for "XINFOTABS: Evaluating Multilingual Tabular Natural Language Inference", containing tables and corresponding hypothesis in 10 languages i.e. English (en), German (de), French (fr), Spanish (es), Afrikaans (af), Russian (ru), Chinese (zh), Arabic (ar), Korean (ko) and Hindi (hi).


# Citation

```
@inproceedings{minhas-etal-2022-xinfotabs,
    title = "{XI}nfo{T}ab{S}: Evaluating Multilingual Tabular Natural Language Inference",
    author = "Minhas, Bhavnick  and
      Shankhdhar, Anant  and
      Gupta, Vivek  and
      Aggarwal, Divyanshu  and
      Zhang, Shuo",
    booktitle = "Proceedings of the Fifth Fact Extraction and VERification Workshop (FEVER)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.fever-1.7",
    doi = "10.18653/v1/2022.fever-1.7",
    pages = "59--77",
    abstract = "The ability to reason about tabular or semi-structured knowledge is a fundamental problem for today{'}s Natural Language Processing (NLP) systems. While significant progress has been achieved in the direction of tabular reasoning, these advances are limited to English due to the absence of multilingual benchmark datasets for semi-structured data. In this paper, we use machine translation methods to construct a multilingual tabular NLI dataset, namely XINFOTABS, which expands the English tabular NLI dataset of INFOTABS to ten diverse languages. We also present several baselines for multilingual tabular reasoning, e.g., machine translation-based methods and cross-lingual. We discover that the XINFOTABS evaluation suite is both practical and challenging. As a result, this dataset will contribute to increased linguistic inclusion in tabular reasoning research and applications.",
}
```

# Data

The data is categorized according to language having tables and hypothesis files for each language in seperate folder. 

The data split is as follows:

```
data/
├── af/
│   ├── af_tables/
│   │   ├── af_T0.json
│   │   ├── af_T1.json
│   │   ├── af_T10.json
│   │   ├── af_T100.json
│   │   └── ...
│   ├── af_hypothesis_alpha1.csv
│   ├── af_hypothesis_alpha2.csv
│   ├── af_hypothesis_alpha3.csv
│   ├── af_hypothesis_dev.csv
│   └── af_hypothesis_train.csv
├── ar/
│   └── ...
├── de/
│   └── ...
├── en/
│   └── ...
├── es/
│   └── ...
├── fr/
│   └── ...
├── hi/
│   └── ...
├── ko/
│   └── ...
├── ru/
│   └── ...
└── zh/
    └── ...
```


