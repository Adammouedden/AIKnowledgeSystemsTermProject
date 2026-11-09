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
        self._first_obs = self.obs
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


def _snapshot(e: AlfEnv) -> dict:
    return dict(gamefile=e.gamefile, max_steps=e.max_steps, task=e.task, obs=e.obs, admissible=list(e.admissible),
                _first_obs=getattr(e, "_first_obs", ""),
                carrying=list(e.carrying), t=e.t, won=e.won, visited=list(e.visited), _anchor=e.anchor())


class EnvView:
    """What the trainer sees of a runner living in a worker process: the same
    attributes `build_prompt` reads, refreshed after every load/step."""

    def __init__(self, d: dict):
        self.__dict__.update(d)

    def anchor(self) -> str:
        return self._anchor


def _worker(conn, n_local, gamefile, max_steps):
    envs = [AlfEnv(gamefile, max_steps) for _ in range(n_local)]
    while True:
        try:
            cmd, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):  # parent exited
            return
        if cmd == "close":
            return
        out = []
        for j, arg in payload:
            if cmd == "load":
                envs[j].load(arg)
                out.append(_snapshot(envs[j]))
            elif cmd == "restore":  # arg = (gamefile, actions): load, then replay the actions verbatim
                gf, acts = arg
                envs[j].load(gf)
                obs, done, won = [], False, False
                for a in acts:
                    o, done, won, _ = envs[j].step(a)
                    obs.append(o)
                    if done:
                        break
                out.append((_snapshot(envs[j]), obs, done, won))
            else:  # step
                _, done, won, valid = envs[j].step(arg)
                out.append((_snapshot(envs[j]), done, won, valid))
        conn.send(out)


class EnvPool:
    """`n` reusable runners spread over `workers` processes.

    TextWorld steps cost ~30 ms of pure-Python CPU each, so stepping 64
    episodes serially took as long as generating their actions on the GPU.
    Runner i lives in worker i % workers and is still reused across games
    (the per-game leak stays bounded per process). Views are plain objects
    with the attributes of AlfEnv, so prompt construction is unchanged.
    """

    def __init__(self, n: int, gamefile: str, max_steps: int, workers: int = 16):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")  # never fork a process that holds CUDA state
        self.n, self.W = n, max(1, min(workers, n))
        self.conns, self.procs = [], []
        for w in range(self.W):
            a, b = ctx.Pipe()
            n_local = len(range(w, n, self.W))
            p = ctx.Process(target=_worker, args=(b, n_local, gamefile, max_steps), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.views = [None] * n

    def __len__(self):
        return self.n

    def _call(self, cmd, items):
        """items: list of (runner index, arg). Returns results in the same order."""
        per = {}
        for i, arg in items:
            per.setdefault(i % self.W, []).append((i, arg))
        for w, lst in per.items():
            self.conns[w].send((cmd, [(i // self.W, arg) for i, arg in lst]))
        res = {}
        for w, lst in per.items():
            for (i, _), r in zip(lst, self.conns[w].recv()):
                res[i] = r
        return [res[i] for i, _ in items]

    def load(self, gamefiles: list) -> list:
        """Load gamefiles[i] into runner i; returns the views of runners 0..len-1."""
        for i, d in enumerate(self._call("load", list(enumerate(gamefiles)))):
            self.views[i] = EnvView(d)
        return self.views[: len(gamefiles)]

    def restore(self, items: list) -> list:
        """items: [(runner, gamefile, actions)]. Loads each game into its runner and
        replays the actions inside the worker. Returns [(obs list, done, won)]."""
        out = []
        for (i, _, _), (d, obs, done, won) in zip(items, self._call("restore", [(i, (g, a)) for i, g, a in items])):
            self.views[i] = EnvView(d)
            out.append((obs, done, won))
        return out

    def step(self, idx: list, actions: list) -> list:
        """Step runner idx[k] with actions[k]; returns [(obs, done, won, valid)] and refreshes views."""
        out = []
        for i, (d, done, won, valid) in zip(idx, self._call("step", list(zip(idx, actions)))):
            self.views[i] = EnvView(d)
            out.append((d["obs"], done, won, valid))
        return out

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass


def make_pool(n: int, gamefile: str, max_steps: int, workers: int = 16) -> EnvPool:
    """A pool of `n` reusable runners, all initially on the same game."""
    return EnvPool(n, gamefile, max_steps, workers)


SYSTEM = (
    "You act in a household. 'go to X' moves to receptacle X (you can only touch where you are); "
    "'open X' reveals a closed receptacle's contents; 'take O from X' picks up O (one at a time); "
    "'move O to X' puts what you carry onto X; 'use X' switches on lamp X.\n"
    "Plan: search receptacles for the object the task names, opening closed ones, take it, go to "
    "the target receptacle, move it there. Never re-check a receptacle you already looked inside.\n"
    "Answer with only the number of your chosen action, e.g. `3`."
)


# Free-form text actions (the paper's setting). Two complete worked
# trajectories, one per task type, given as earlier chat turns in exactly the
# format of a live step -- including the admissible list -- so the model sees
# the action being *copied from the list* every time. Rooms and objects are
# invented, so neither example is a training or evaluation task. A first
# version showed the examples as one transcript without admissible lists; the
# model then copied object names out of the examples (68% invalid, see
# docs/DECISIONS.md).
SYSTEM_TEXT = (
    "You act in a household to complete a task. You can only interact with the receptacle you are at. "
    "Each turn you see the task, recent steps, the current observation and the admissible actions. "
    "Reply with exactly one line from the admissible actions, copied character for character, and nothing else. "
    "Objects and receptacles that are not in the current list do not exist here."
)

_EX = [
    ("put some cellphone on bed", [
        ("You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, "
         "a garbagecan 1, and a sidetable 1.",
         ["go to bed 1", "go to desk 1", "go to garbagecan 1", "go to sidetable 1"], "go to desk 1"),
        ("You arrive at desk 1. On the desk 1, you see a alarmclock 1, and a pen 2.",
         ["go to bed 1", "go to garbagecan 1", "go to sidetable 1", "take alarmclock 1 from desk 1",
          "take pen 2 from desk 1"], "go to sidetable 1"),
        ("You arrive at sidetable 1. On the sidetable 1, you see a cellphone 1, and a keychain 1.",
         ["go to bed 1", "go to desk 1", "go to garbagecan 1", "take cellphone 1 from sidetable 1",
          "take keychain 1 from sidetable 1"], "take cellphone 1 from sidetable 1"),
        ("You pick up the cellphone 1 from the sidetable 1.",
         ["go to bed 1", "go to desk 1", "go to garbagecan 1", "move cellphone 1 to sidetable 1"], "go to bed 1"),
        ("You arrive at bed 1. On the bed 1, you see a pillow 1.",
         ["go to desk 1", "go to garbagecan 1", "go to sidetable 1", "move cellphone 1 to bed 1"],
         "move cellphone 1 to bed 1"),
    ]),
    ("look at book under the desklamp", [
        ("You are in the middle of a room. Looking quickly around you, you see a armchair 1, a drawer 1, "
         "a dresser 1, and a sidetable 1.",
         ["go to armchair 1", "go to drawer 1", "go to dresser 1", "go to sidetable 1"], "go to sidetable 1"),
        ("You arrive at sidetable 1. On the sidetable 1, you see a desklamp 1, and a tissuebox 1.",
         ["go to armchair 1", "go to drawer 1", "go to dresser 1", "take tissuebox 1 from sidetable 1",
          "use desklamp 1"], "go to drawer 1"),
        ("You arrive at drawer 1. The drawer 1 is closed.",
         ["go to armchair 1", "go to dresser 1", "go to sidetable 1", "open drawer 1"], "open drawer 1"),
        ("You open the drawer 1. The drawer 1 is open. In it, you see a book 2.",
         ["go to armchair 1", "go to dresser 1", "go to sidetable 1", "take book 2 from drawer 1"],
         "take book 2 from drawer 1"),
        ("You pick up the book 2 from the drawer 1.",
         ["go to armchair 1", "go to dresser 1", "go to sidetable 1", "move book 2 to drawer 1"], "go to sidetable 1"),
        ("You arrive at sidetable 1. On the sidetable 1, you see a desklamp 1, and a tissuebox 1.",
         ["go to armchair 1", "go to drawer 1", "go to dresser 1", "move book 2 to sidetable 1", "use desklamp 1"],
         "use desklamp 1"),
    ]),
]


# The paper's own prompt: verl-agent's ALFWorld template (Feng et al.;
# agent_system/environments/prompts/alfworld.py), reproduced verbatim. The
# model reasons inside <think></think> and answers inside <action></action>;
# the whole response, reasoning included, is the step's action a_t whose mean
# token NLL is GACA's H_t.
VERL_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

VERL_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def build_think_prompt(env, history: list, history_len: int = 2) -> list:
    """verl-agent's ALFWorld prompt; `history` is a list of (action, obs) pairs,
    where obs is what that action produced. The step-0 prompt uses the no-history
    template, as verl-agent does."""
    adm = ", ".join(f"'{a}'" for a in env.admissible)
    if not history:
        obs = f"{env.obs} Your task is to: {env.task}." if env.task else env.obs
        return [{"role": "user", "content": VERL_TEMPLATE_NO_HIS.format(current_observation=obs, admissible_actions=adm)}]
    start = max(0, len(history) - history_len)
    # verl-agent's memory pairs each observation with the action taken from it.
    obs_seq = [env._first_obs if hasattr(env, "_first_obs") else ""] + [o for _, o in history]
    parts = []
    for k in range(start, len(history)):
        parts.append(f"\n[Observation {k + 1}: '{obs_seq[k]}', Action {k + 1}: '{history[k][0]}']")
    return [{"role": "user", "content": VERL_TEMPLATE.format(
        task_description=env.task, step_count=len(history), history_length=len(history) - start,
        action_history="".join(parts), current_step=len(history) + 1, current_observation=env.obs,
        admissible_actions=adm)}]


_ACT_RE = re.compile(r"<action>(.*?)</action>", re.S | re.I)


def parse_think_action(text: str, admissible: list):
    """verl-agent's projection: valid only with both <think> and <action> tags
    and an action that, normalized, is admissible. Returns (action, valid)."""
    m = _ACT_RE.findall(text)
    if "<think>" not in text or "</think>" not in text or not m:
        return "noop", False
    cand = _norm_act(m[-1])
    for a in admissible:
        if _norm_act(a) == cand:
            return a, True
    return cand or "noop", False


def _text_turn(task, now, adm, hist):
    lines = [f"Task: {task}"]
    if hist:
        lines.append("Recent steps:")
        lines += [f"  {i}. {a} -> {o}" for i, (a, o) in enumerate(hist, start=1)]
    lines.append(f"Now: {now}")
    lines.append("Admissible actions:\n" + "\n".join(adm))
    lines.append("Your action:")
    return "\n".join(lines)


def _fewshot_messages(max_hist: int = 3) -> list:
    msgs = []
    for task, traj in _EX:
        hist = []
        for now, adm, act in traj:
            msgs.append({"role": "user", "content": _text_turn(task, now, adm, hist[-max_hist:])})
            msgs.append({"role": "assistant", "content": act})
            hist.append((act, now))
    return msgs


FEWSHOT = _fewshot_messages()


def build_prompt(env: AlfEnv, history: list, history_len: int = 6, obs_chars: int = 110, mode: str = "index") -> list:
    if mode == "think":
        return build_think_prompt(env, history)
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
    if mode == "text":
        lines.append("Admissible actions:\n" + "\n".join(env.admissible))
        lines.append("Your action:")
        return [{"role": "system", "content": SYSTEM_TEXT}, *FEWSHOT, {"role": "user", "content": "\n".join(lines)}]
    lines.append("Actions: " + "; ".join(f"{i}={a}" for i, a in enumerate(env.admissible, start=1)))
    lines.append("Your choice (number only):")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


_NUM_RE = re.compile(r"\d+")


def _norm_act(s: str) -> str:
    return " ".join(s.lower().replace("`", " ").split()).strip(" .\"'")


def parse_text_action(text: str, admissible: list):
    """Free-form mode. Returns (action, exact).

    `exact` is True only when the reply, up to case/whitespace/trailing
    punctuation, *is* an admissible action. Anything else is sent to the
    environment verbatim and therefore counts as inadmissible -- no fuzzy
    rescue, so the invalid-action rate measured here is the real one.
    """
    first = text.strip().splitlines()[0] if text.strip() else ""
    first = re.sub(r"^(your action|action)\s*:\s*", "", first.strip(), flags=re.I)
    cand = _norm_act(first)
    for a in admissible:
        if _norm_act(a) == cand:
            return a, True
    return cand or "noop", False


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
