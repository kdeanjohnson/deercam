"""Provider-neutral prompts and output schemas for the vision LLM."""
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeVar

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

Unknown = Literal["unknown"]


class Rack(BaseModel):
    points_left: int = Field(description="Best-estimate points on the deer's left antler incl. the main beam tip "
                                         "(count tines over ~1 inch); -1 only if that antler is not visible at all")
    points_right: int = Field(description="Best-estimate points on the deer's right antler incl. the main beam tip "
                                          "(count tines over ~1 inch); -1 only if that antler is not visible at all")
    points_confidence: Literal["high", "medium", "low"] = Field(
        description="How sure the point counts are given angle, overlap and image quality")
    frame: Literal["typical", "non_typical", "unknown"]
    spread: Literal["inside_ears", "at_ears", "outside_ears", "unknown"] = Field(
        description="Inside spread relative to the ear tips with ears in alert position")
    mass: Literal["thin", "average", "heavy", "unknown"]
    tine_length: Literal["short", "medium", "long", "unknown"]
    brow_tines: Literal["both", "left_only", "right_only", "none", "unknown"]
    beam_shape: str = Field(description="Short description of main beam curve, e.g. 'wide sweeping, tips curl in'")
    distinctive_features: list[str] = Field(
        description="Features that would tell this buck apart: kickers, drop tines, split brows, broken tines, "
                    "stickers, unusual asymmetry. Say which side (deer's left/right) and which tine.")


class DeerAnalysis(BaseModel):
    # Field order matters: the model fills these in sequence, so it has to look at the head
    # and say what is (or isn't) there before it commits to a sex.
    species: Literal["deer", "other", "unknown"]
    other_species: str = Field(description="If species is 'other', its common name, e.g. 'wild turkey', 'feral hog', "
                                           "'coyote', 'bobcat', 'raccoon', 'armadillo', 'domestic dog', 'crow'; else empty")
    head_visible: bool = Field(description="Is the top of this deer's head (between and above the ears) in frame and discernible?")
    antler_evidence: str = Field(
        description="Describe exactly what is on top of the head between/above the ears, e.g. 'two main beams "
                    "with 3 upright tines each', 'bare head, ears only', 'head down in grass, can't see'. Do not "
                    "count fence wire, branches, feeder legs or another deer's ears as antlers.")
    antlers_visible: Literal["yes", "no", "cannot_tell"]
    sex: Literal["buck", "doe", "fawn", "unknown"] = Field(
        description="buck only if antlers_visible is 'yes'; doe/fawn only if the head is visible and bare; "
                    "otherwise unknown")
    view: Literal["front", "left_side", "right_side", "rear", "quartering", "head_down", "partial"]
    image_quality: Literal["good", "fair", "poor"]
    age_class: Literal["fawn", "yearling", "2.5", "3.5", "4.5_plus", "unknown"]
    rack: Rack | None = Field(description="Only when sex is buck; null otherwise")
    body_marks: list[str] = Field(description="Scars, notched/torn ears, limps, coat marks, unusual body shape")
    summary: str = Field(description="One sentence a hunter would use to describe this deer")


class AntlerCheck(BaseModel):
    """Second look at first-pass buck calls, which mistake does for bucks about half the time."""
    animal: str = Field(description="What the main (largest, most central) animal in the crop is")
    head_location: str = Field(description="Where the main deer's head is in the crop and what it is doing")
    above_head: str = Field(description="Describe literally what is directly above/between the ears of the main "
                                        "deer's head, and what is behind it in the background")
    lookalikes: str = Field(description="Any fence wire, feeder legs, branches, grass, ears or legs of other deer "
                                        "near the head that could be mistaken for antlers")
    antlers: Literal["yes", "no", "unclear"] = Field(
        description="'yes' only if bony antler beams/tines clearly grow from the main deer's own skull")


class WildlifeCheck(BaseModel):
    """Second look at first-pass 'not a deer' calls, with the whole photo for context. The first look sees
    only the crop and misnames a lot: turkeys' barred feathers as skunks, armadillos or zebras."""
    looks: str = Field(description="What the boxed animal looks like: feathers, fur or armor; body shape; legs; "
                                   "head, snout and ears; tail. Only what you can actually see")
    others_in_photo: str = Field(description="The other animals in the whole photo, if any")
    animal: str = Field(description="Common name of the boxed animal, e.g. 'wild turkey', 'feral hog', 'bobcat', "
                                    "'raccoon', 'nine-banded armadillo', 'crow'; 'white-tailed deer' if it is a "
                                    "deer; 'unknown' if you can't tell")
    confidence: Literal["high", "medium", "low"]


class CandidateComparison(BaseModel):
    label: str
    same_features: list[str] = Field(description="Antler features visible in both that agree")
    conflicting_features: list[str] = Field(
        description="Antler features clearly visible in both that disagree (not just hidden by angle)")


class MatchDecision(BaseModel):
    # Look first, decide after: the model fills these in order.
    new_photo_rack: str = Field(description="What you can actually see of the NEW PHOTO's rack: beam shape, "
                                            "tine count/length/position per side, spread vs ears, abnormal points")
    comparisons: list[CandidateComparison]
    match: str = Field(description="The candidate label (e.g. 'A') of the same individual buck, or 'none'")
    confidence: float = Field(description="0.0-1.0 confidence that the match (or 'none') is correct")
    reasoning: str = Field(description="Which antler features agree or disagree")


class PairCheck(BaseModel):
    """One candidate, for a person comparing side by side: look first, then say."""
    new_photo_rack: str = Field(description="What you can actually see of the NEW PHOTO's rack, per side")
    candidate_rack: str = Field(description="What the candidate's reference photos show of his rack, per side")
    same_features: list[str] = Field(description="Antler features visible in both that agree")
    conflicting_features: list[str] = Field(
        description="Antler features clearly visible in both that disagree (not just hidden by angle)")
    verdict: Literal["same", "different", "cant_tell"] = Field(
        description="'same' only when specific features line up and nothing conflicts")
    confidence: float = Field(description="0.0-1.0 confidence in the verdict")
    explanation: str = Field(description="Two or three plain sentences for the hunter: what decided it")


ANALYZE_SYSTEM = (
    "You analyze trail-camera crops of animals for a deer hunter in {region}. Photos are low resolution "
    "and often night infrared (grayscale), blurry, or partially out of frame. Most deer at these feeders are "
    "does and fawns, so only call a buck when you can actually see antlers; spikes and small forks count. "
    "Once you've established it is a buck, read the rack the way an experienced hunter would: give a "
    "best-estimate point count and use the confidence field to say how sure you are. Left/right always means "
    "the deer's own left/right, not the viewer's."
)

ANALYZE_PROMPT = (
    "This crop is from a trail camera named '{camera}' taken at {when}. Classify the animal and, if it is a "
    "buck, describe the rack in enough detail to recognize this individual buck again in other photos."
)

class NameIdea(BaseModel):
    name: str = Field(description="The buck's name, 1-3 words")
    why: str = Field(description="A few words on what in the photo or context inspired it")


class NameIdeas(BaseModel):
    ideas: list[NameIdea] = Field(description="Six distinct name ideas")


NAME_SYSTEM = (
    "You help a family of deer hunters come up with nicknames for individual whitetail bucks they watch on "
    "trail cameras, the kind of names used around deer camp: 'Big Ten', 'Lefty', 'Crab Claw', 'Pond Buck', "
    "'Ghost', 'Ol' Wide'. The best names come from something recognizable about the buck: rack shape or "
    "point count, a distinctive tine, body or coat, behavior, or where and when he shows up. Keep them "
    "short and easy to say. Mix a couple of plain descriptive ones with a few that have more personality. "
    "Don't reuse a name that's already taken."
)

VERIFY_SYSTEM = (
    "You check trail-camera crops that an automatic classifier flagged as an antlered buck. That classifier "
    "is wrong about half the time: it mistakes does for bucks, reading ears, fence wire, feeder legs, "
    "branches, or other deer's body parts as antlers, and it sometimes flags turkeys or other animals. Most "
    "deer at these feeders are does. Your job is to confirm or overturn the call by looking only at the main "
    "deer's own head. Antlers are hard, branched or spike-shaped structures growing up from the skull "
    "between and slightly in front of the ears; even small spikes stick up visibly above the hairline. Ears "
    "are soft, broad, and come in a pair at the sides. Photos are low resolution; if you cannot see the top "
    "of the head, answer 'unclear'."
)

VERIFY_PROMPT = "Does the main deer in this crop have antlers?"

WILDLIFE_SYSTEM = (
    "You check trail-camera photos from {region}. A quick first pass said the animal in the red box is not a "
    "deer and guessed what it is from a small crop; those guesses are often wrong. Name the animal in the "
    "red box from its own features: feathers and a beak, or fur; two legs or four; the shape of the head, "
    "ears and tail. Use the whole photo for its size against the feeder, fence and other animals. Common "
    "mix-ups: a turkey's barred feathers read as a skunk's stripes or an armadillo's bands; a bobcat's "
    "spotted coat, rounded head and short tail missed for a hog's; a crow or vulture in flight blurred into "
    "something big. Animals seen here include white-tailed deer, wild turkeys, feral hogs, raccoons, "
    "opossums, nine-banded armadillos, coyotes, bobcats, foxes, skunks, rabbits, squirrels, crows, vultures "
    "and other birds, cattle, dogs and cats. Say 'unknown' rather than guess."
)

WILDLIFE_PROMPT = "What animal is in the red box? Trail camera '{camera}', {when}. The first pass said: '{first}'."

MATCH_SYSTEM = (
    "You re-identify individual whitetail bucks across trail-camera photos by comparing antler structure: "
    "point count per side, tine lengths and positions, main beam curve, spread, mass, brow tines, and "
    "abnormal points. Antlers are the same within a season, so a clear structural difference (e.g. a "
    "kicker present on one, absent on the other with the same side visible) rules a match out. Pose and "
    "lighting differ between photos; account for points hidden by angle. Judge from the images themselves, "
    "since body size and coat change little between bucks and the rack is what identifies them. Several "
    "different bucks use the same feeders, so similar-looking typical racks are common. The crops are "
    "scaled to different sizes, so judge rack size only relative to the same deer's own ears, head and "
    "body (tine height vs. ear length, spread vs. the ear tips, beam length vs. the head), never by how "
    "big it looks in the picture; a young buck with a small rack is not the same deer as a mature one with "
    "a big rack even if both are 8 points. Pick a candidate "
    "only when specific features line up; if unsure, answer 'none' with low confidence. If the NEW PHOTO "
    "shows no antlers, or too little of them to compare, answer 'none'."
)


RETURNING_SYSTEM = (
    "You help a deer hunter recognize whitetail bucks returning from earlier seasons on trail cameras. "
    "Bucks cast their antlers every winter and grow a new rack each year, so the NEW PHOTO's rack will "
    "not be identical to the candidates' older photos. A maturing buck usually gains mass, tine length, "
    "spread and sometimes points until about 5.5 years old, and may shrink when old or injured. What "
    "tends to carry over from year to year is the rack's basic architecture: brow tine shape and length, "
    "how the tines are spaced and angled, the curve of the main beams and whether the tips turn in, "
    "the spread relative to the body, a tendency toward forked G2s, splits or kickers in the same spot, "
    "and overall typical vs. non-typical character. Non-rack clues also carry over: face and throat "
    "patch markings, notched or torn ears, scars, body shape, and which camera he uses. Left/right "
    "always means the deer's own left/right. Photos are low resolution. Several bucks share these "
    "feeders and similar racks are common, so pick a candidate only when several of these persistent "
    "traits line up and nothing rules him out; otherwise answer 'none'. Your answer is a suggestion "
    "for the hunter to check, so say which traits you relied on.\n\n"
    "Some candidates come with the owner's notes on non-antler marks (e.g. a torn left ear). Use a "
    "note ONLY to rule that candidate out when the NEW PHOTO clearly shows the opposite; never count "
    "it as evidence for a match, since such marks are easy to imagine in a low-resolution photo."
)


@dataclass
class ImagePart:
    """An image plus a caption that precedes it in the prompt."""
    caption: str
    path: Path
    box: tuple[float, float, float, float] | None = None  # normalized x1, y1, x2, y2 to outline in red


def encode_jpeg(path: Path, max_side: int = 1024, min_side: int = 640, box=None) -> str:
    """Crops from 1024x768 originals are often ~150px; vision models read antlers far better upscaled."""
    img = Image.open(path).convert("RGB")
    if box:
        x1, y1, x2, y2 = box
        ImageDraw.Draw(img).rectangle((x1 * img.width, y1 * img.height, x2 * img.width, y2 * img.height),
                                      outline=(255, 0, 0), width=max(3, img.width // 250))
    scale = min(max(min_side / min(img.size), 1.0), max_side / max(img.size))
    if scale != 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


class VisionProvider(Protocol):
    name: str

    def structured(self, system: str, images: list[ImagePart], prompt: str, schema: type[T]) -> T: ...
