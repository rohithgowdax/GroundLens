# PFL-DocVQA Dataset

PFL-DocVQA is a dataset that we use for additional experiments. We define how we setup the dataset

1. Download the dataset from: https://benchmarks.elsa-ai.eu/?ch=2&com=tasks
    - Pick Blue Team setting
    - Login to the task page
    - Click Download
2. Run the pipeline to create question priors with the following command (uses `vidore/colSmol-500M`):
```bash
python3 -m src.scripts.generation.gen_question_prior \
    --pipeline pfl_docvqa \
    --pfl_data_dir /path/to/PFL-DocVQA/
```
3. Use the dataset for training or inference. The folder should now look like:
```
PFL-DocVQA/
├── images/
│   └── *.jpg
├── imdb_val.npy
├── imdb_train_client_0.npy
├── ...
├── imdb_train_client_9.npy
└── q_priors/
    ├── train/
    │   └── {question_id}.pt
    └── val/
        └── {question_id}.pt
```

## Answer bboxes

Answer bounding boxes (`answer_bbox`) are computed automatically at dataset load time in `PFLDocVQA._compute_answer_bbox()`. The function matches answer strings against the OCR word tokens (`ocr_normalized_boxes`) to find the tightest bounding span, then expands it by 10% width / 15% height. The collate functions (`pfl_docvqa_with_q_prior_collate_fn`) simply read the pre-computed `sample.answer_bbox` field. No separate generation step is required.

