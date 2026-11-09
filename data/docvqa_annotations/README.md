# GroundLens Answer Bounding Box Annotations

Manual bounding box annotations for answer regions in the DocVQA dataset.
300 examples were annotated in total: 200 train and 100 validation.

## Background

The base dataset includes an `answer_ocr_bbox` field that locates the answer text region
on each document image. After running a pipeline to extract answer locations with OCR, some examples could not
be resolved and were left with `null`. This annotation effort fills in those missing
bounding boxes to investigate whether providing the missing supervision signal leads to
a performance gain during training.

| File                           | Description                                                        |
| ------------------------------ | ------------------------------------------------------------------ |
| `annotated_bboxes.json`        | Raw export from Label Studio                                       |
| `label_studio_train.json`      | Label Studio task import file for the train split (200 tasks)      |
| `label_studio_validation.json` | Label Studio task import file for the validation split (100 tasks) |

---

## Annotation Process

Examples where `answer_ocr_bbox` was `null` were identified and selected — 200 from the
train split and 100 from the validation split. The corresponding document images were
saved locally and served over a local HTTP server so Label Studio could load them.

A Label Studio project was set up with the following labeling config:

```xml
<View>
  <View style="margin-bottom: 10px;">
    <Text name="qid" value="ID: $question_id"/>
    <Text name="question" value="Question: $question"/>
    <Text name="answer" value="Answer: $answer"/>
  </View>

  <Image name="image" value="$image"/>

  <RectangleLabels name="label" toName="image">
    <Label value="Answer" background="red"/>
  </RectangleLabels>
</View>
```

Each task displayed the question ID, question, and expected answer above the document
image. I drew a tight bounding box around the text region that answers the question
using the **Answer** label.

After all 300 tasks were completed, annotations were exported from Label Studio as
`annotated_bboxes.json`.

---

## Annotation Format

Label Studio stores coordinates as percentages of the image dimensions:

```json
{
  "question_id": 18612,
  "question": "What is the CSF?",
  "answer": "721",
  "label": [
    {
      "x": 19.54,
      "y": 36.9,
      "width": 12.84,
      "height": 5.68,
      "original_width": 1695,
      "original_height": 2167,
      "rectanglelabels": ["Answer"]
    }
  ]
}
```

These are converted to normalized `[x1, y1, x2, y2]` coordinates in `[0, 1]` before
use in training.
