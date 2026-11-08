"""ALFWorld (TextWorld) wrapper + prompt construction.

One AlfEnv instance == one TextWorld game (one task). Deterministic given the
gamefile, so G rollouts of the same task all start from the same state, which is
what GiGPO/GACA anchor-state grouping needs.
"""

import glob
import hashlib
import json
import os
import re

ALFWORLD_DATA = os.environ.setdefault(
    "ALFWORLD_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/alfworld")
)

import textworld  # noqa: E402
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler  # noqa: E402

TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

_INFOS = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
# Commands that are never required to solve any ALFWorld task type. Dropping
# them shortens the prompt and shrinks the action space the policy must cover.
_DROP = {"help", "inventory", "look"}
_DROP_PREFIX = ("examine ", "close ")


def _split_dir() -> dict:
    return {
        "train": os.path.join(ALFWORLD_DATA, "json_2.1.1", "train"),
        "valid_seen": os.path.join(ALFWORLD_DATA, "json_2.1.1", "valid_seen"),
        "valid_unseen": os.path.join(ALFWORLD_DATA, "json_2.1.1", "valid_unseen"),
    }


def list_games(split: str, task_types=None, cache_dir=None) -> list:
    """Solvable, non-movable-receptacle game files for a split, optionally filtered by task type."""
    cache_dir = cache_dir or os.path.join(ALFWORLD_DATA, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"games_{split}.json")
    if os.path.exists(cache):
        games = json.load(open(cache))
    else:
        games = []
        for gf in sorted(glob.glob(os.path.join(_split_dir()[split], "*", "*", "game.tw-pddl"))):
            if "movable_recep" in gf or "Sliced" in gf:
                continue
            try:
                data = json.load(open(gf))
            except Exception:
                continue
            if not data.get("solvable", False):
                continue
            tt = json.load(open(os.path.join(os.path.dirname(gf), "traj_data.json")))["task_type"]
            games.append({"gamefile": gf, "task_type": tt})
        json.dump(games, open(cache, "w"))
    if task_types:
        games = [g for g in games if g["task_type"] in task_types]
    return games


def _clean(obs: str) -> str:
    obs = re.sub(r"-=.*?=-", "", obs)
    return " ".join(obs.split()).strip()


class AlfEnv:
    """One reusable ALFWorld task runner.

    TextWorld interns game logic in process-global caches, so constructing a
    fresh environment per task leaks a few MB that no amount of closing or
    garbage collection gives back. One environment object reused via load()
    costs about a fifth of that, so the trainer keeps a fixed pool of these and
    loads a new game into each instead of building new ones.
    """

    def __init__(self, gamefile: str, max_steps: int = 30):
        self._env = textworld.start(gamefile, request_infos=_INFOS, wrappers=[AlfredDemangler])
        self.gamefile = gamefile
        self.max_steps = max_steps
        self.task = ""
        self.obs = ""
        self.admissible = []
        self.carrying = []
        self.t = 0
        self.won = False
        self.visited = []

    def _parse(self, state):
        obs = _clean(state.feedback)
        m = re.search(r"Your task is to:\s*(.*)", obs)
        if m:
            self.task = m.group(1).strip().rstrip(".")
            obs = obs[: m.start()].strip()
        self.obs = obs
        self.admissible = [
            c for c in state.admissible_commands if c not in _DROP and not c.startswith(_DROP_PREFIX)
        ]
        # Objects the agent is holding: exactly those it can "move ... to ...".
        self.carrying = sorted({c.split(" to ")[0][len("move ") :] for c in self.admissible if c.startswith("move ")})
        self.won = bool(state.won)

    def load(self, gamefile: str):
        """Switch this runner to another task and reset it."""
        if gamefile != self.gamefile:
            self._env.load(gamefile)
            self.gamefile = gamefile
        return self.reset()

    def reset(self):
        self.t = 0
        self.task = ""
        self.visited = []
        self._parse(self._env.reset())
        return self.obs

    def step(self, action: str):
        """Returns (obs, done, won, valid). `valid` is False for a non-admissible action."""
        valid = action in self.admissible
        if action.startswith("go to"):
            where = action[len("go to ") :]
            if where not in self.visited:
                self.visited.append(where)
        state, _, done = self._env.step(action)
        self.t += 1
        self._parse(state)
        done = bool(done) or self.won or self.t >= self.max_steps
        return self.obs, done, self.won, valid

    def anchor(self) -> str:
        """Hash of the current environment state: observation + the admissible set.

        The admissible set encodes inventory and open/closed status, so this
        distinguishes states that happen to share an observation string.
        """
        key = self.obs + "||" + "|".join(sorted(self.admissible))
        return hashlib.md5(key.encode()).hexdigest()[:16]


def make_pool(n: int, gamefile: str, max_steps: int) -> list:
    """A pool of `n` reusable runners, all initially on the same game."""
    return [AlfEnv(gamefile, max_steps) for _ in range(n)]


SYSTEM = (
    "You act in a household. 'go to X' moves to receptacle X (you can only touch where you are); "
    "'open X' reveals a closed receptacle's contents; 'take O from X' picks up O (one at a time); "
    "'move O to X' puts what you carry onto X; 'use X' switches on lamp X.\n"
    "Plan: search receptacles for the object the task names, opening closed ones, take it, go to "
    "the target receptacle, move it there. Never re-check a receptacle you already looked inside.\n"
    "Answer with only the number of your chosen action, e.g. `3`."
)


def build_prompt(env: AlfEnv, history: list, history_len: int = 6, obs_chars: int = 110) -> list:
    """Chat messages for the current step. `history` is a list of (action, obs) pairs.

    Kept deliberately compact: every step re-encodes this prompt, so its length
    sets the cost of both the rollout prefill and the policy update.
    """
    lines = [f"Task: {env.task}", f"Step {env.t + 1}/{env.max_steps}."]
    if env.visited:
        lines.append("Already visited: " + ", ".join(env.visited))
    if history:
        lines.append("Recent steps:")
        start = max(0, len(history) - history_len)
        for i, (act, obs) in enumerate(history[start:], start=start + 1):
            lines.append(f"  {i}. {act} -> {obs[:obs_chars]}")
    lines.append(f"Carrying: {', '.join(env.carrying) if env.carrying else 'nothing'}")
    lines.append(f"Now: {env.obs or 'nothing new'}")
    lines.append("Actions: " + "; ".join(f"{i}={a}" for i, a in enumerate(env.admissible, start=1)))
    lines.append("Your choice (number only):")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


_NUM_RE = re.compile(r"\d+")


def parse_action(text: str, admissible: list) -> str:
    """Map the model's reply to an admissible action.

    Primary path: the reply is just an integer, read as a 1-based index into the
    admissible list. Otherwise: the longest admissible action appearing verbatim
    in the reply, else the first integer in the reply. Returns "noop" when
    nothing matches, which the trainer counts as an invalid action.
    """

    def by_index(tok):
        i = int(tok) - 1
        return admissible[i] if 0 <= i < len(admissible) else None

    bare = text.strip().strip(".*`#:()[]").strip()
    if bare.isdigit():
        return by_index(bare) or "noop"
    low = " ".join(text.split()).lower()
    hits = [a for a in admissible if a in low]
    if hits:
        return max(hits, key=len)
    m = _NUM_RE.search(text)
    if m:
        return by_index(m.group()) or "noop"
    return "noop"


def probe_games(games: list, max_steps: int = 2) -> list:
    """Annotate each game with its number of admissible actions at reset.

    Uses a single reusable runner, so the cost is ~1 MB per distinct game rather
    than ~5 MB. Results are cached by the set of game files requested.
    """
    cache_dir = os.path.join(ALFWORLD_DATA, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5("|".join(sorted(g["gamefile"] for g in games)).encode()).hexdigest()[:16]
    cache = os.path.join(cache_dir, f"probe_{key}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    out = []
    runner = None
    for g in games:
        try:
            if runner is None:
                runner = AlfEnv(g["gamefile"], max_steps)
                runner.reset()
            else:
                runner.load(g["gamefile"])
            out.append(dict(g, n_adm0=len(runner.admissible), task=runner.task))
        except Exception:
            continue
    json.dump(out, open(cache, "w"))
    return out
