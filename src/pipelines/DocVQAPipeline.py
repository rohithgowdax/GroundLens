from datasets import load_dataset
from paddleocr import PaddleOCR
import numpy as np
import logging
from datetime import datetime, timezone
from rapidfuzz.distance import Levenshtein
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re
import unicodedata


# Constants
SP_DOCVQA_DATASET = "pixparse/docvqa-single-page-questions"
NEW_SP_DOCVQA_DATASET = "indrehus/docvqa-single-page-questions"


# Helper function to render an example with bounding boxes
def render_example(
    example,
    figsize=(10, 12),
    dpi=150,
    save_path=None,
):
    img = example["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")

    # Draw PaddleOCR bbox if available
    paddle_bbox = example.get("paddle_ocr_bbox")
    if paddle_bbox is not None:
        x1 = paddle_bbox[0] * W
        y1 = paddle_bbox[1] * H
        w = (paddle_bbox[2] - paddle_bbox[0]) * W
        h = (paddle_bbox[3] - paddle_bbox[1]) * H
        ax.add_patch(Rectangle((x1, y1), w, h, linewidth=3, edgecolor="tab:blue", facecolor=(1, 0, 0, 0.18)))

    # Draw Textract bbox if available
    textract_bbox = example.get("textract_ocr_bbox")
    if textract_bbox is not None:
        x1 = textract_bbox[0] * W
        y1 = textract_bbox[1] * H
        w = (textract_bbox[2] - textract_bbox[0]) * W
        h = (textract_bbox[3] - textract_bbox[1]) * H
        ax.add_patch(Rectangle((x1, y1), w, h, linewidth=3, edgecolor="tab:green", facecolor=(1, 0, 0, 0.18)))

    # Draw final answer bbox if available
    rect_norm = example.get("answer_ocr_bbox")
    if rect_norm is not None:
        x1 = rect_norm[0] * W
        y1 = rect_norm[1] * H
        w = (rect_norm[2] - rect_norm[0]) * W
        h = (rect_norm[3] - rect_norm[1]) * H
        ax.add_patch(Rectangle((x1, y1), w, h, linewidth=3, edgecolor="tab:red", facecolor=(1, 0, 0, 0.18)))

    # Set title with question and answer
    q = str(example.get("question", "")).strip()
    a = str(example.get("answer") or (example.get("answers") or [""])[0]).strip()
    color_descriptions = "PaddleOCR: Blue, TextractOCR: Green, Final Answer Location: Red\n"
    ax.set_title(f"{color_descriptions}Q: {q}\nA: {a}", loc="left", fontsize=11)

    # Save or show
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    else:
        plt.show()
    plt.close(fig)


class TextMatcher:
    """Utility class for matching answer text to OCR results and calculating confidence scores."""
    def __init__(self):
        self.fuzzy_thresh = 0.82
        self.digits_thresh = 0.95

    @staticmethod
    def to_lower(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    @staticmethod
    def norm_text(s: str) -> str:
        s = unicodedata.normalize("NFKC", str(s or ""))
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return " ".join(s.split())

    @staticmethod
    def digits(s: str) -> str:
        return re.sub(r"\D+", "", str(s or ""))

    def is_answer_in_text(self, answer: str, text: str):
        return self.to_lower(answer) in self.to_lower(text)

    def get_question_anchor_distance(self, question: str, text: str) -> int:
        return Levenshtein.distance(self.to_lower(question), self.to_lower(text))

    def get_sim_ratio(self, a: str, b: str) -> float:
        a = a or ""
        b = b or ""
        if not a and not b:
            return 1.0
        denom = max(len(a), len(b), 1)
        return 1.0 - (Levenshtein.distance(a, b) / denom)

    def get_best_match(self, texts, answer: str):
        ans_norm = self.norm_text(answer)
        ans_digits = self.digits(answer)

        best = {"score": -1.0, "idx": None, "line": None}
        best_d = {"score": -1.0, "idx": None, "line": None}
        exact_norm_idx = None
        exact_digits_idx = None
        substr_norm_idx = None
        substr_digits_idx = None

        for i, line in enumerate(texts or []):
            s = str(line or "")
            n = self.norm_text(s)
            d = self.digits(s)

            if ans_norm and ans_norm == n and exact_norm_idx is None:
                exact_norm_idx = i
            if ans_digits and ans_digits == d and exact_digits_idx is None:
                exact_digits_idx = i
            if ans_norm and ans_norm in n and substr_norm_idx is None:
                substr_norm_idx = i
            if ans_digits and ans_digits in d and substr_digits_idx is None:
                substr_digits_idx = i

            if ans_norm:
                r = self.get_sim_ratio(ans_norm, n)
                if r > best["score"]:
                    best = {"score": r, "idx": i, "line": s}
            if ans_digits:
                rd = self.get_sim_ratio(ans_digits, d)
                if rd > best_d["score"]:
                    best_d = {"score": rd, "idx": i, "line": s}

        if exact_norm_idx is not None:
            reason = "exact_norm"
            best_idx = exact_norm_idx
            contains = True
        elif exact_digits_idx is not None:
            reason = "exact_digits"
            best_idx = exact_digits_idx
            contains = True
        elif substr_norm_idx is not None:
            reason = "substring_norm"
            best_idx = substr_norm_idx
            contains = True
        elif substr_digits_idx is not None:
            reason = "substring_digits"
            best_idx = substr_digits_idx
            contains = True
        elif best["score"] >= self.fuzzy_thresh:
            reason = "fuzzy_norm"
            best_idx = best["idx"]
            contains = True
        elif best_d["score"] >= self.digits_thresh:
            reason = "fuzzy_digits"
            best_idx = best_d["idx"]
            contains = True
        else:
            reason = "none"
            best_idx = None
            contains = False

        return {
            "contains": contains,
            "reason": reason,
            "best_idx": best_idx,
            "best_norm": best,
            "best_digits": best_d,
            "ans_norm": ans_norm,
            "ans_digits": ans_digits,
        }

    @staticmethod
    def calculate_confidence(m: Dict):
        if not isinstance(m, dict):
            return -1.0
        base = max(m.get("best_norm", {}).get("score", -1.0),
                   m.get("best_digits", {}).get("score", -1.0))
        reason = (m.get("reason") or "none")
        boost = 0.25 if reason.startswith("exact") else (0.10 if reason.startswith("substring") else 0.0)
        return base + boost


class BoundingBoxHandler:
    """Utility class for normalizing and expanding bounding boxes."""
    def __init__(self, answer_bbox_x_expansion, answer_bbox_y_expansion):
        self.answer_bbox_x_expansion = answer_bbox_x_expansion
        self.answer_bbox_y_expansion = answer_bbox_y_expansion


    def to_list_of_four(self, bbox:list):
        if len(bbox) == 4: 
            return bbox
        
        if len(bbox) == 6: 
            # turn into 4
            return bbox
        
        return bbox

    def expand_bbox(self, bbox: list):
        assert len(bbox) == 4, "expanding bbox requires list of four values"
        x1, y1, x2, y2 = map(float, bbox)
        width = max(1e-6, x2 - x1)
        height = max(1e-6, y2 - y1)
        x_expansion = width * self.answer_bbox_x_expansion
        y_expansion = height * self.answer_bbox_y_expansion
        new_x1 = max(0.0, x1 - x_expansion)
        new_y1 = max(0.0, y1 - y_expansion)
        new_x2 = min(1.0, x2 + x_expansion)
        new_y2 = min(1.0, y2 + y_expansion)
        return [new_x1, new_y1, new_x2, new_y2]


    def normalize_bbox(self, bbox, W: int, H: int):
      """
      Normalize to [x1, y1, x2, y2] in [0,1].

      Accepts:
        - [x1, y1, x2, y2]  (pixels or already normalized)
        - [x1,y1,x2,y2,x3,y3,x4,y4]  (pixels or already normalized)

      Returns None for any other shape.
      """
      if bbox is None:
        return None

      # Turn numpy array to list of cords
      if type(bbox) == np.ndarray:
          bbox = bbox.tolist()

      # helper
      def _clamp01(v): 
          return max(0.0, min(1.0, float(v)))

      if len(bbox) == 4:
          x1, y1, x2, y2 = map(float, bbox)
          x1, x2 = x1 / W, x2 / W
          y1, y2 = y1 / H, y2 / H

          # enforce ordering and clamp
          x1, x2 = min(x1, x2), max(x1, x2)
          y1, y2 = min(y1, y2), max(y1, y2)
          return [_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)]

      if len(bbox) == 8:
          pts = list(map(float, bbox))
          xs, ys = pts[0::2], pts[1::2]
          # if looks like pixels, convert points
          if max(xs + ys) > 1.0001:
              xs = [x / W for x in xs]
              ys = [y / H for y in ys]
          x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
          return [_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)]

      print("bbox was neither", ", ".join(map(str, bbox)))
      return None



class DocVQAPipeline:
    def __init__(
        self,
        ds_name: str = SP_DOCVQA_DATASET,
        new_ds_name: str = NEW_SP_DOCVQA_DATASET, 
        dataset=None,
        ocr_model: PaddleOCR = None,
        device: str = "cpu",
        hf_token: str = "",
        answer_bbox_x_expansion: float = 0.10,
        answer_bbox_y_expansion: float = 0.15,
        textract_min_fuzzy_min: float = 0.60,
        paddle_min_fuzzy_min: float = 0.60,
        verbose: bool = False,
        cache_dir:str = None, 
    ):
        assert ds_name == SP_DOCVQA_DATASET, f"Dataset {ds_name} not supported. Only {SP_DOCVQA_DATASET} is supported."
        assert hf_token != "", "HuggingFace token must be provided"

        # Load dataset
        self.dataset = load_dataset(ds_name, cache_dir=cache_dir) if dataset is None else dataset
        self.dataset_name = ds_name
        self.push_repo_id = f"{new_ds_name}-answer-ocr"
        self.hf_token = hf_token

        # Load OCR model
        self.ocr_model = PaddleOCR(use_angle_cls=False, lang="en") if ocr_model is None else ocr_model
        self.device = device
        self.verbose = verbose

        # Store fuzzy find min 
        self.textract_min_fuzzy_min = textract_min_fuzzy_min
        self.paddle_min_fuzzy_min = paddle_min_fuzzy_min

        # Init boundary box handler 
        self.bbox_handler = BoundingBoxHandler(
            answer_bbox_x_expansion=answer_bbox_x_expansion,
            answer_bbox_y_expansion=answer_bbox_y_expansion,
        )

        # Init text matcher
        self.text_matcher = TextMatcher()

    def build_commit_info(self)-> tuple[str, str]:
        # Get time of run
        now = datetime.now(timezone.utc)
        ts_human = now.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Try to get the sizes and splits
        splits = []
        split_sizes = {}
        total = 0
        try:
            splits = list(self.dataset.keys())
            split_sizes = {s: len(self.dataset[s]) for s in splits}
            total = sum(split_sizes.values())
        except Exception:
            pass

        commit_description = (
            f"Run: {ts_human}\n"
            f"Timestamp (UTC): {ts_human}\n"
            f"Dataset: {getattr(self, 'dataset_name', 'unknown')}\n"
            f"Splits: {', '.join(splits) if splits else '(none)'}\n"
            f"Split sizes:\n{split_sizes}\n"
            f"Total examples: {total}\n"
        )

        commit_message = f"[RUN] docvqa bbox answer prior run, {ts_human} for splits {', '.join(splits) if splits else '(none)'}."

        return commit_message, commit_description

    def get_answer_bbox_from_paddle_ocr(
        self,
        answer: str,
        paddle_ocr_results: Any,
        logger: Optional[logging.Logger] = None,
        idx: Optional[int] = None,
    ):
        if not answer or paddle_ocr_results is None:
            return None, {"reason": "none", "contains": False}

        texts = paddle_ocr_results.get("rec_texts")
        boxes = paddle_ocr_results.get("rec_boxes")

        match = self.text_matcher.get_best_match(texts, answer)
        chosen_idx = match["best_idx"]
        bbox = boxes[chosen_idx] if (chosen_idx is not None and chosen_idx < len(boxes)) else None

        if self.verbose and logger:
            logger.info(
                f"[idx={idx}] Paddle match: contains={'YES' if match['contains'] else 'NO'} "
                f"({match['reason']}); best_norm={match['best_norm']['score']:.3f}, "
                f"best_digits={match['best_digits']['score']:.3f}"
            )
            if chosen_idx is not None:
                logger.info(
                    f"[idx={idx}] Paddle chose idx={chosen_idx}; text={texts[chosen_idx]!r}; bbox={bbox}"
                )
        return bbox, match

    def get_answer_bbox_from_amazon_textract(
        self,
        answer: str,
        question: str,
        ocr_results: dict,
        logger: Optional[logging.Logger] = None,
        idx: Optional[int] = None,
    ):
        if self.verbose and logger:
            logger.info(f"[idx={idx}] Collecting lines from Amazon Textract")


        # Collect all lines 
        lines = (ocr_results or {}).get("lines") or []
        if not lines:
            if self.verbose and logger:
                logger.info(f"[idx={idx}] Textract has no lines.")
            return None, {"reason": "none", "contains": False}


        # Collect all text with their respective bboxes
        texts, boxes = [], []
        for ln in lines:
            texts.append(ln.get("text"))
            boxes.append(ln.get("bounding_box"))

        match = self.text_matcher.get_best_match(texts, answer)
        if self.verbose and logger:
            logger.info(f"[idx={idx}] Textract best-match summary: {match}")

        chosen_idx = match["best_idx"]

        # Only run tiebreaker if match already "contains" the answer (i.e., it’s a valid match class)
        if chosen_idx is None and match.get("contains"):
            if self.verbose and logger:
                logger.info(f"[idx={idx}] Using question-anchor tiebreaker for Textract")
            norm_idx = match["best_norm"]["idx"]
            digits_idx = match["best_digits"]["idx"]
            candidates = []
            if norm_idx is not None:
                candidates.append(norm_idx)
            if digits_idx is not None and digits_idx != norm_idx:
                candidates.append(digits_idx)
            if candidates:
                best_id = None
                best_anchor = float("inf")
                for i in candidates:
                    d = self.text_matcher.get_question_anchor_distance(question or "", texts[i] or "")
                    if d < best_anchor:
                        best_anchor = d
                        best_id = i
                chosen_idx = best_id

        bbox = boxes[chosen_idx] if (chosen_idx is not None and chosen_idx < len(boxes)) else None

        if self.verbose and logger:
            logger.info(
                f"[idx={idx}] Textract match: contains={'YES' if match['contains'] else 'NO'} "
                f"({match['reason']}); best_norm={match['best_norm']['score']:.3f}, "
                f"best_digits={match['best_digits']['score']:.3f}"
            )
            if chosen_idx is not None:
                logger.info(
                    f"[idx={idx}] Textract chose idx={chosen_idx}; text={texts[chosen_idx]!r}; bbox={bbox}"
                )
            else:
                logger.debug(f"[idx={idx}] Textract: no candidates for answer={answer!r}")
        return bbox, match

    def is_acceptable(self, m: Dict, which: str) -> bool:
        if not isinstance(m, dict) or not m.get("contains"):
            return False
        r = m.get("reason", "none")
        if r.startswith("exact") or r.startswith("substring"):
            return True
        
        # fuzzy needs to clear a floor
        best = max(m.get("best_norm", {}).get("score", -1.0),
                m.get("best_digits", {}).get("score", -1.0))
        floor = self.textract_min_fuzzy_min if which == "textract" else self.paddle_min_fuzzy_min
        return best >= floor

    def select_best_ocr_match(
        self,
        question: str,
        paddle_bbox: Optional[list],
        textract_bbox: Optional[list],
        paddle_match: Optional[Dict],
        textract_match: Optional[Dict],
        logger: Optional[logging.Logger] = None,
        idx: Optional[int] = None,
    ):  
        # Get paddle and textract match 
        pm = paddle_match or {"contains": False, "reason": "none"}
        tm = textract_match or {"contains": False, "reason": "none"}

        # Calculate their confidence 
        p_conf = self.text_matcher.calculate_confidence(pm)
        t_conf = self.text_matcher.calculate_confidence(tm)

        tex_ok = (textract_bbox is not None) and self.is_acceptable(tm, "textract")
        pad_ok = (paddle_bbox is not None)   and self.is_acceptable(pm, "paddle")

        if tex_ok:
            # Use textract if available 
            src, bbox, conf, match = "textract", textract_bbox, t_conf, tm
            why = "prefer_textract"
        elif pad_ok:
            # Use paddle if available 
            src, bbox, conf, match = "paddle", paddle_bbox, p_conf, pm
            why = "fallback_paddle"
        else:
            # Neither was available 
            src, bbox, conf, match = "none", None, -1.0, None
            why = "no_acceptable_match"

        if logger and self.verbose:
            t_best = max((tm or {}).get("best_norm", {}).get("score", -1.0),
                        (tm or {}).get("best_digits", {}).get("score", -1.0))
            p_best = max((pm or {}).get("best_norm", {}).get("score", -1.0),
                        (pm or {}).get("best_digits", {}).get("score", -1.0))
            logger.info(f"[idx={idx}] SELECTOR | why={why} | src={src.upper()} | "
                        f"tex_ok={tex_ok} (best={t_best:.3f}) | pad_ok={pad_ok} (best={p_best:.3f}) | "
                        f"tex_bbox={textract_bbox} | pad_bbox={paddle_bbox} | "
                        f"tex_conf={t_conf} | pad_conf={p_conf}")

        return src, bbox, conf, match, p_conf, t_conf

    # ---- core ----
    def process_example(
        self, example: Dict[str, Any], logger: Optional[logging.Logger] = None, idx: Optional[int] = None
    ):
        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        W, H = img.size

        answer = example.get("answers")[0]
        question = example.get("question", "")
        ocr_results = example.get("ocr_results") or {}

        if self.verbose and logger:
            logger.info(f"[idx={idx}] Image size: {W}x{H}")

        # Get the textract first 
        textract_raw, textract_match = self.get_answer_bbox_from_amazon_textract(
            answer, question, ocr_results, logger=logger, idx=idx
        )
        textract_bbox = self.bbox_handler.normalize_bbox(textract_raw, W, H)

        if self.verbose and logger:
            logger.info(f"[idx={idx}] Textract raw: {textract_raw}")
            logger.info(f"[idx={idx}] Textract norm: {textract_bbox}")

        # Decide whether we can skip PaddleOCR
        tex_ok = (textract_bbox is not None) and self.is_acceptable(textract_match, "textract")

        # Skip paddleOCR if it works well
        if tex_ok:
            if self.verbose and logger:
                logger.info(f"[idx={idx}] Skipping PaddleOCR: acceptable Textract match found.")
            paddle_raw = None
            paddle_bbox = None
            paddle_match = {
                "contains": False,
                "reason": "skipped",
                "best_idx": -1,
                "best_norm": {"score": -1.0, "idx": -1, "line": ""},
                "best_digits": {"score": -1.0, "idx": -1, "line": ""},
                "ans_norm": "",
                "ans_digits": "",
            }
        else:
            # Run with Paddle OCR 
            if self.verbose and logger:
                logger.info(f"[idx={idx}] Running PaddleOCR: Textract not acceptable.")
            # Convert image only if we actually run Paddle
            image_np = np.array(img)
            paddle_ocr_result = self.ocr_model.predict(
                image_np,
                text_det_limit_side_len=1280,
                text_det_limit_type="max",
                text_det_box_thresh=0.40,
                text_det_unclip_ratio=1.6,
                text_rec_score_thresh=0.25,
                return_word_box=False,
            )[0]

            if self.verbose and logger:
                paddle_line_count = len((paddle_ocr_result or {}).get("rec_texts") or [])
                logger.info(f"[idx={idx}] Paddle OCR lines: {paddle_line_count}")
                logger.info(f"[idx={idx}] Q={question!r} | A={answer!r} | Textract lines: {len((ocr_results or {}).get('lines', []) or [])}")

            paddle_raw, paddle_match = self.get_answer_bbox_from_paddle_ocr(
                answer, paddle_ocr_result, logger=logger, idx=idx
            )
            paddle_bbox = self.bbox_handler.normalize_bbox(paddle_raw, W, H)

            if self.verbose and logger:
                logger.info(f"[idx={idx}] Paddle raw: {paddle_raw}")
                logger.info(f"[idx={idx}] Paddle norm: {paddle_bbox}")

        # If a bbox failed, say why/what we saw (verbose only)
        if self.verbose and logger:
            if paddle_bbox is None and 'paddle_raw' in locals() and paddle_raw is not None:
                arr = np.array(paddle_raw, dtype=object)
                logger.debug(f"[idx={idx}] Paddle bbox failed to normalize. "
                            f"type={type(paddle_raw).__name__}, shape={getattr(arr, 'shape', None)}, "
                            f"value_sample={str(paddle_raw)[:120]}...")
            if textract_bbox is None and textract_raw is not None:
                arr = np.array(textract_raw, dtype=object)
                logger.debug(f"[idx={idx}] Textract bbox failed to normalize. "
                            f"type={type(textract_raw).__name__}, shape={getattr(arr, 'shape', None)}, "
                            f"value_sample={str(textract_raw)[:120]}...")

        # Save intermediate boxes
        example["paddle_ocr_bbox"] = paddle_bbox
        example["textract_ocr_bbox"] = textract_bbox

        # Final selection (policy prefers Textract if acceptable)
        final_src, final_bbox, final_conf, _, paddle_conf, textract_conf = self.select_best_ocr_match(
            question=question,
            paddle_bbox=paddle_bbox,
            textract_bbox=textract_bbox,
            paddle_match=paddle_match,
            textract_match=textract_match,
            logger=logger,
            idx=idx,
        )

        # Expand the final bbox
        if final_bbox is not None:
            final_bbox = self.bbox_handler.expand_bbox(final_bbox)

        example["answer_ocr_bbox"] = final_bbox
        example["answer_match_debug"] = {
            "source": final_src,
            "final_conf": final_conf,
            "paddle_conf": paddle_conf,
            "textract_conf": textract_conf,
            "paddle_match": paddle_match,
            "textract_match": textract_match,
            "paddle_bbox_none": paddle_bbox is None,
            "textract_bbox_none": textract_bbox is None,
        }

        if self.verbose and logger:
            logger.info(f"[idx={idx}] SELECTED source={final_src.upper()} | bbox={final_bbox} | "
                        f"paddle_conf={paddle_conf:.3f} | textract_conf={textract_conf:.3f}")
            render_example(example)

        return example


    def process_dataset(
        self, logger: logging.Logger, max_examples: int = -1, log_interval: int = 100, private: bool = True
    ):

        # Empty match debug, which is just all fields nulled out
        def _default_match_debug():
            # All fields have fixed types: floats, ints, bools, strings
            return {
                "source": "",                      # string
                "final_conf": -1.0,                # float64
                "paddle_conf": -1.0,               # float64
                "textract_conf": -1.0,             # float64
                "paddle_bbox_none": True,          # bool
                "textract_bbox_none": True,        # bool
                "paddle_match": {
                    "contains": False,             # bool
                    "reason": "",                  # string
                    "best_idx": -1,                # int64
                    "best_norm": {"score": -1.0, "idx": -1, "line": ""},  # float64, int64, string
                    "best_digits": {"score": -1.0, "idx": -1, "line": ""},# float64, int64, string
                    "ans_norm": "",                # string
                    "ans_digits": "",              # string
                },
                "textract_match": {
                    "contains": False,
                    "reason": "",
                    "best_idx": -1,
                    "best_norm": {"score": -1.0, "idx": -1, "line": ""},
                    "best_digits": {"score": -1.0, "idx": -1, "line": ""},
                    "ans_norm": "",
                    "ans_digits": "",
                },
            }
        

        # Go through each split
        for split in list(self.dataset.keys()):
            logger.info(f"Processing split: {split}")
            ds = self.dataset[split]

            # Limit per split if requested
            if isinstance(max_examples, int) and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))

            total = len(ds)
            counter = {"n": 0}

            # For split, we dont need OCR results, so we can skip this section
            if split == "test":
                def _fill_test(ex):
                    ex = dict(ex)
                    ex["paddle_ocr_bbox"] = []
                    ex["textract_ocr_bbox"] = []
                    ex["answer_ocr_bbox"] = []
                    ex["answer_match_debug"] = _default_match_debug()
                    return ex

                # Fill with empty 
                ds = ds.map(
                    _fill_test,
                    with_indices=False,
                    num_proc=1,
                    desc="Populate empty OCR fields (test)",
                    batched=False,
                    keep_in_memory=False,
                    load_from_cache_file=False,
                )
                self.dataset[split] = ds
                logger.info(f"Finished processing split: {split} (skipped OCR)")
                continue
            

            # Method to give the map function to process the split and log it 
            def map_example(ex, idx):
                # If answers are missing/empty (e.g., test split), skip OCR and fill deterministic defaults
                answers = ex.get("answers") or []
                if not answers:
                    # Likely "test" split, so we dont process the data
                    ex = dict(ex)  # copy to avoid mutating original reference
                    ex["paddle_ocr_bbox"] = []
                    ex["textract_ocr_bbox"] = []
                    ex["answer_ocr_bbox"] = []
                    ex["answer_match_debug"] = _default_match_debug()
                else:
                    # Process the example
                    ex = self.process_example(ex, logger=logger, idx=idx)

                    # Remove the paddle ocr results from the row
                    ex.pop("paddle_ocr_results", None)

                counter["n"] += 1
                if counter["n"] % max(1, log_interval) == 0:
                    logger.info(f"[{split}] processed {counter['n']}/{total}")
                return ex

            # Map with minimal memory pressure and no stale caches
            ds = ds.map(
                map_example,
                with_indices=True,
                num_proc=1,
                desc=f"Answer bboxes ({split})",
                batched=False,
                keep_in_memory=False,
                load_from_cache_file=False,
            )

            # Make sure paddle_ocr_results are removed
            transient_cols = [c for c in ["paddle_ocr_results"] if c in ds.column_names]
            if transient_cols:
                ds = ds.remove_columns(transient_cols)

            # Set the new mapped ds split to the built in dataset 
            self.dataset[split] = ds
            logger.info(f"Finished processing split: {split}")

        # Casting 
        ref_split = "train" if "train" in self.dataset else next(iter(self.dataset.keys()))
        ref_features = self.dataset[ref_split].features
        logger.info(f"Using '{ref_split}' as reference schema for casting.")

        # Cast all splits to the same schema. Make sure all splits contain the same features 
        self.dataset = self.dataset.cast(ref_features) 
        for split, ds in self.dataset.items():
            logger.info(f"[{split}] features AFTER casting:\n{ds.features}")


        # Set test split to None for answer bbox since we dont have the answers for the test set
        logger.info("Clearing answer priors for text")
        def wipe_bboxes(ex):
            ex["answer_ocr_bbox"] = None          
            return ex

        self.dataset["test"] = self.dataset["test"].map(
            wipe_bboxes,
            desc="Zeroing bbox columns on test",
            batched=False,
            load_from_cache_file=False,
            keep_in_memory=False,
        )

        # Push to Hub with a simple, informative commit
        logger.info(f"Pushing to HF repo: {self.push_repo_id} (private={private})")
        commit_message, commit_description = self.build_commit_info()
        self.dataset.push_to_hub(
            self.push_repo_id,
            token=self.hf_token,
            private=private,
            commit_message=commit_message,
            commit_description=commit_description,
        )
        logger.info(f"Saved processed data on HF: {self.push_repo_id}")
