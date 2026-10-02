"""
RetinaBot - a small rule-based assistant embedded in the demo application.

Why rule-based?
    The assistant must work offline, cost nothing, never hallucinate medical
    facts and be fully auditable.  Every answer below is written by hand from
    the International Clinical Diabetic Retinopathy (ICDR) severity scale and
    standard screening guidance, and the bot only ever *explains*; it never
    diagnoses.

How it works
    1. The user message is lower-cased and tokenised.
    2. Every intent has a list of keyword groups; the intent whose keywords
       overlap the message the most wins (ties -> first defined).
    3. Answers can reference the *context* (the last prediction made in the
       app) so the bot can say "your image was graded Moderate NPDR ...".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Clinical reference content (ICDR scale + common screening intervals)
# --------------------------------------------------------------------------- #
STAGE_INFO: Dict[int, Dict[str, str]] = {
    0: {
        "name": "No DR",
        "findings": "no visible abnormalities.",
        "advice": "routine repeat screening in about 12 months, plus good control of blood sugar, blood pressure and cholesterol.",
        "urgency": "Routine",
    },
    1: {
        "name": "Mild non-proliferative DR",
        "findings": "microaneurysms only (tiny bulges in the retinal capillaries).",
        "advice": "repeat screening in 6-12 months; no treatment is usually needed at this stage, but tighter diabetes control slows progression.",
        "urgency": "Routine",
    },
    2: {
        "name": "Moderate non-proliferative DR",
        "findings": "haemorrhages, hard exudates or cotton-wool spots - more than microaneurysms but less than severe NPDR.",
        "advice": "referral to an ophthalmologist for a full dilated examination, typically within 3-6 months; check for diabetic macular oedema.",
        "urgency": "Referable",
    },
    3: {
        "name": "Severe non-proliferative DR",
        "findings": "extensive haemorrhages in all four quadrants, venous beading or intraretinal microvascular abnormalities (the 4-2-1 rule).",
        "advice": "urgent referral to an ophthalmologist, normally within 4 weeks; the risk of progressing to proliferative DR within a year is high.",
        "urgency": "Urgent referral",
    },
    4: {
        "name": "Proliferative DR",
        "findings": "growth of new fragile blood vessels (neovascularisation) and/or vitreous or pre-retinal haemorrhage.",
        "advice": "immediate referral - ideally within 1-2 weeks - because laser photocoagulation or anti-VEGF injections may be needed to prevent vision loss.",
        "urgency": "Immediate referral",
    },
}

DISCLAIMER = ("Remember: this tool is a screening aid built for a university coursework. "
              "It is not a medical device and cannot replace an examination by an eye-care professional.")


@dataclass
class ChatContext:
    """What the app knows about the last prediction (may be empty)."""

    grade: Optional[int] = None
    confidence: Optional[float] = None
    probabilities: Optional[List[float]] = None
    model_name: str = "EfficientNetB0"
    metrics: Dict[str, float] = field(default_factory=dict)
    external_metrics: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Intents
# --------------------------------------------------------------------------- #
@dataclass
class Intent:
    name: str
    keywords: List[List[str]]   # a list of synonym groups; each matched group scores 1
    handler: str                # name of the method on RetinaBot that builds the reply


INTENTS: List[Intent] = [
    Intent("greeting", [["hello", "hi", "hey", "good morning", "good evening", "start"]], "greeting"),
    Intent("thanks", [["thank", "thanks", "cheers", "great help"]], "thanks"),
    Intent("result", [["my result", "my image", "my scan", "what does", "mean", "diagnos", "grade", "prediction", "predicted", "my stage", "this stage", "the result"]], "result"),
    Intent("severity", [["serious", "be worried", "should i worry", "worried", "worry", "worse", "dangerous", "danger", "bad", "scared", "afraid",
                          "concern", "go blind", "blindness", "lose my sight", "emergency"]], "severity"),
    Intent("next_steps", [["next", "should i", "what now", "do now", "refer", "referral", "see a doctor", "ophthalmolog", "appointment", "urgent", "treatment", "treat"]], "next_steps"),
    Intent("what_is_dr", [["what is diabetic retinopathy", "what is dr", "retinopathy", "explain dr", "define", "disease"]], "what_is_dr"),
    Intent("stages", [["stage", "stages", "grade", "grades", "levels", "scale", "classification", "categories", "icdr"]], "stages"),
    Intent("symptoms", [["symptom", "signs", "feel", "blurry", "blur", "vision", "floaters", "notice"]], "symptoms"),
    Intent("risk", [["risk", "cause", "why", "prevent", "avoid", "reduce", "control", "sugar", "hba1c", "blood pressure", "lifestyle"]], "risk"),
    Intent("confidence", [["confidence", "confident", "sure", "probability", "certain", "reliable", "trust"]], "confidence"),
    Intent("model", [["model", "how does", "how do you", "work", "cnn", "efficientnet", "network", "algorithm", "transfer learning", "ai", "trained"]], "model"),
    Intent("gradcam", [["heatmap", "heat map", "grad", "cam", "red area", "highlight", "colour", "color", "overlay", "explain the image"]], "gradcam"),
    Intent("accuracy", [["accuracy", "accurate", "performance", "f1", "kappa", "sensitivity", "specificity", "metrics", "how good"]], "accuracy"),
    Intent("dataset", [["dataset", "data", "kaggle", "aptos", "images used", "training data", "where from"]], "dataset"),
    Intent("limitations", [["limit", "limitation", "wrong", "mistake", "error", "fail", "disclaimer", "safe", "medical device", "replace"]], "limitations"),
    Intent("help", [["help", "what can you", "options", "questions", "ask"]], "help"),
]


class RetinaBot:
    """Tiny intent-matching assistant.  ``reply(message, context)`` returns text."""

    def __init__(self) -> None:
        self.intents = INTENTS

    # ------------------------------------------------------------------ #
    def match(self, message: str) -> Optional[Intent]:
        text = " " + re.sub(r"[^a-z0-9 ]", " ", message.lower()) + " "
        best, best_score = None, 0
        for intent in self.intents:
            score = 0
            for group in intent.keywords:
                if any(self._found(kw, text) for kw in group):
                    score += 1
            # small bonus for longer / more specific keywords
            longest = max((len(kw) for group in intent.keywords for kw in group if self._found(kw, text)), default=0)
            score = score * 100 + longest
            if score > best_score:
                best, best_score = intent, score
        return best if best_score >= 100 else None

    @staticmethod
    def _found(keyword: str, text: str) -> bool:
        """Keywords of up to 3 letters must be whole words; longer ones may be word stems ("ophthalmolog")."""
        return f" {keyword} " in text if len(keyword) <= 3 else keyword in text

    def reply(self, message: str, ctx: Optional[ChatContext] = None) -> str:
        ctx = ctx or ChatContext()
        intent = self.match(message)
        if intent is None:
            return self.fallback(ctx)
        return getattr(self, intent.handler)(ctx)

    # ------------------------------------------------------------------ #
    # handlers
    # ------------------------------------------------------------------ #
    def greeting(self, ctx: ChatContext) -> str:
        return ("Hello! I am RetinaBot, the assistant of this diabetic retinopathy screening demo. "
                "Ask me what your result means, what the five DR stages are, what to do next, "
                "or how the AI model works.")

    def thanks(self, ctx: ChatContext) -> str:
        return "You are welcome. Take care of your eyes - and keep your regular diabetes check-ups!"

    def help(self, ctx: ChatContext) -> str:
        return ("You can ask me things like:\n"
                "- *What does my result mean?*\n"
                "- *What should I do next?*\n"
                "- *What are the stages of diabetic retinopathy?*\n"
                "- *What is the heat-map showing?*\n"
                "- *How accurate is the model?*\n"
                "- *How can I reduce my risk?*")

    def result(self, ctx: ChatContext) -> str:
        if ctx.grade is None:
            return ("No image has been graded yet. Upload a fundus photograph in the **Screen a photo** tab "
                    "and I will explain the result.")
        s = STAGE_INFO[ctx.grade]
        conf = f" with {ctx.confidence:.0%} confidence" if ctx.confidence is not None else ""
        txt = (f"Your image was graded **{ctx.grade} - {s['name']}**{conf}. "
               f"At this stage a clinician would expect to see {s['findings']} "
               f"Recommended action: {s['advice']}")
        if ctx.probabilities:
            second = sorted(range(5), key=lambda i: -ctx.probabilities[i])[1]
            if ctx.probabilities[second] > 0.2:
                txt += (f"\n\nThe model also gave {ctx.probabilities[second]:.0%} to "
                        f"**{STAGE_INFO[second]['name']}**, so the image sits close to the boundary between the two grades.")
        return txt + "\n\n" + DISCLAIMER

    def severity(self, ctx: ChatContext) -> str:
        if ctx.grade is None:
            return ("Once a photograph has been checked I can tell you how serious the result is. In general, grades "
                    "0-1 are not urgent, grade 2 needs an appointment with an eye specialist, and grades 3-4 need "
                    "urgent specialist care.")
        s = STAGE_INFO[ctx.grade]
        level = {0: "reassuring: no signs of diabetic retinopathy were found",
                 1: "an early, mild stage that usually needs no treatment, only closer monitoring",
                 2: "not an emergency, but you should see an eye specialist",
                 3: "serious: the eye should be seen by a specialist urgently",
                 4: "very serious: the eye needs specialist care as soon as possible"}[ctx.grade]
        return (f"Your result, **{s['name']}**, is {level}. Recommended action: {s['advice']} Treated in time, "
                f"most sight loss from diabetic retinopathy can be prevented.\n\n{DISCLAIMER}")

    def next_steps(self, ctx: ChatContext) -> str:
        if ctx.grade is None:
            return ("Once an image has been graded I can give stage-specific guidance. In general: grades 0-1 mean "
                    "routine annual screening, grade 2 means referral to an ophthalmologist, and grades 3-4 need "
                    "urgent specialist care.")
        s = STAGE_INFO[ctx.grade]
        return (f"For **{s['name']}** the usual recommendation is: {s['advice']} "
                f"Urgency level: **{s['urgency']}**.\n\n{DISCLAIMER}")

    def what_is_dr(self, ctx: ChatContext) -> str:
        return ("Diabetic retinopathy (DR) is damage to the tiny blood vessels of the retina - the light-sensitive "
                "layer at the back of the eye - caused by long-term high blood sugar. Vessels leak, close off, or "
                "grow abnormally, and without treatment this is a leading cause of blindness in working-age adults. "
                "It is usually symptom-free in the early stages, which is why regular photographic screening matters.")

    def stages(self, ctx: ChatContext) -> str:
        lines = ["The International Clinical DR scale has five grades:"]
        for g, s in STAGE_INFO.items():
            lines.append(f"- **{g} - {s['name']}**: {s['findings']}")
        lines.append("Grades 2 and above are called *referable DR* because the patient should be seen by an eye specialist.")
        return "\n".join(lines)

    def symptoms(self, ctx: ChatContext) -> str:
        return ("Early DR usually causes **no symptoms** at all. Later signs can include blurred or fluctuating vision, "
                "dark spots or 'floaters', difficulty seeing at night, and sudden vision loss if a haemorrhage occurs. "
                "Because the early stages are silent, screening photographs like the one you uploaded are the main way "
                "DR is caught in time.")

    def risk(self, ctx: ChatContext) -> str:
        return ("The main drivers of DR are how long someone has had diabetes and how well blood sugar is controlled. "
                "High blood pressure, high cholesterol, kidney disease, pregnancy and smoking all increase the risk. "
                "Keeping HbA1c, blood pressure and lipids in target range, not smoking, and attending yearly eye "
                "screening are the most effective ways to slow progression.")

    def confidence(self, ctx: ChatContext) -> str:
        if ctx.confidence is None:
            return ("Confidence is the softmax probability the model assigns to its chosen grade. Values above ~80 % "
                    "indicate a clear-cut image; values near 40-50 % mean the image lies between two grades and "
                    "should be reviewed by a person.")
        level = "high" if ctx.confidence >= 0.8 else "moderate" if ctx.confidence >= 0.55 else "low"
        return (f"The model's confidence for your image is **{ctx.confidence:.0%}**, which is {level}. "
                + ("A clear result." if level == "high" else
                   "The image sits near a grade boundary, so a human grader should double-check it."))

    def model(self, ctx: ChatContext) -> str:
        return (f"Under the hood this app uses a **{ctx.model_name}** convolutional neural network pretrained on "
                "ImageNet and fine-tuned on retinal photographs (transfer learning). Each image is first cropped to "
                "the retina, padded to a square and resized to 224x224 (extra contrast filters were tested but did "
                "not help), then the network outputs a probability for each of the five DR grades. Training used data augmentation, "
                "class weighting for the rare severe grades, early stopping and a two-phase fine-tuning schedule.")

    def gradcam(self, ctx: ChatContext) -> str:
        return ("The heat-map is a **Grad-CAM** visualisation. It back-propagates the score of the predicted grade to "
                "the last convolutional layer and colours the regions that pushed the decision: red/yellow areas "
                "contributed most, blue areas hardly at all. For a trustworthy prediction the warm colours should "
                "sit on lesions (haemorrhages, exudates, abnormal vessels) rather than on the image border or the "
                "optic disc alone.")

    def accuracy(self, ctx: ChatContext) -> str:
        m = ctx.metrics or {}
        if m:
            parts = []
            if "accuracy" in m:
                parts.append(f"accuracy {m['accuracy']:.1%}")
            qwk = m.get("qwk", m.get("quadratic_weighted_kappa"))
            if qwk is not None:
                parts.append(f"quadratic weighted kappa {qwk:.3f}")
            if "macro_f1" in m:
                parts.append(f"macro F1 {m['macro_f1']:.3f}")
            if "referable_sensitivity" in m:
                parts.append(f"referable-DR sensitivity {m['referable_sensitivity']:.1%}")
            if "referable_specificity" in m:
                parts.append(f"referable-DR specificity {m['referable_specificity']:.1%}")
            text = ("On the held-out test set (15 % of the APTOS 2019 images, never used for training or tuning) the "
                    "model achieved " + ", ".join(parts) + ".")
            ext = ctx.external_metrics or {}
            ext_qwk = ext.get("qwk", ext.get("quadratic_weighted_kappa"))
            if ext_qwk is not None:
                text += (f" On 35,126 images from a completely different dataset (EyePACS 2015, USA) it reached "
                         f"accuracy {ext.get('accuracy', float('nan')):.1%}, quadratic weighted kappa {ext_qwk:.3f} and "
                         f"referable-DR sensitivity {ext.get('referable_sensitivity', float('nan')):.1%}, which shows how "
                         "well it generalises to other cameras and populations.")
            return text + (" The neighbouring grades (mild vs moderate, severe vs proliferative) are the hardest to "
                           "separate, which is also true for human graders.")
        return ("Performance is measured on a held-out test split with accuracy, precision, recall, F1-score and the "
                "quadratic weighted kappa. See the **About** tab for the exact numbers of the loaded model.")

    def dataset(self, ctx: ChatContext) -> str:
        return ("The model was trained on the **APTOS 2019 Blindness Detection** dataset from Kaggle (Aravind Eye "
                "Hospital, India): 3,662 colour fundus photographs graded 0-4 by clinicians, using the version resized "
                "to 224x224 pixels (kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data). Duplicate "
                "photographs were detected and removed before splitting. The data is imbalanced - about half of the "
                "images show no DR and only 5 % show severe NPDR. The model was then tested, without any retraining, "
                "on 35,126 images of the EyePACS 2015 dataset from the USA (external validation).")

    def limitations(self, ctx: ChatContext) -> str:
        return ("Limitations to keep in mind: the model was trained on ~3,600 images from one hospital network, so "
                "it may perform worse on other cameras or populations; it cannot detect diabetic macular oedema or "
                "other eye diseases; low-quality or badly-centred photographs reduce accuracy; and adjacent grades "
                "are often confused. " + DISCLAIMER)

    def fallback(self, ctx: ChatContext) -> str:
        return ("I am not sure I understood that. " + self.help(ctx))
