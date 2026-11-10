# Video narration script (~3.5 minutes, ~520 words)

Cues for what to show are in brackets. To trim to 3 minutes, drop the GACA-noise sentence and the positive-only sentence.

---

**[Title page]**

Hi, I'm Adam. This project asks a simple question about training language-model agents with reinforcement learning: when an agent makes a decision, where inside its response is that decision actually made?

**[Background, Figure 1]**

Agents like this are trained on long tasks with a single reward at the end. In ALFWorld, our benchmark, the agent has to complete household tasks like "put a keychain on the desk" over up to fifty steps. The hard part is credit assignment: which actions deserve credit for success?

GRPO, the standard method, gives every step of an episode the same credit. GiGPO improves on that by comparing different rollouts that reach the same state, which gives each step its own advantage. GACA, an ICLR 2026 paper, mixes those two signals per step, using the model's uncertainty to decide how much to trust each one.

But all of these methods stop at the step. Each response is about a hundred tokens of reasoning followed by a short action, and every one of those tokens gets the same credit.

**[Section 6, the flip test]**

So before training anything, we measured. We cut real responses at different points, resampled the rest, and checked whether the action changed. That tells us exactly where the decision gets settled.

We found three things. First, the decision is made in the reasoning: 82 percent of it is settled before the action is even written. Second, uncertainty doesn't find it. The final tenth of a response holds 19 percent of the decision but only 3 percent of the entropy. Third, a cheap probe does find it. At each cut we force the reasoning to close and ask how likely the final action already is. Weighting tokens by how much that probability rises puts 1.65 times more credit on decisive reasoning than uniform credit does. Attention, by comparison, manages only 1.12 times.

We also found that GACA's own uncertainty score is about 85 percent noise from whichever reasoning path happened to be sampled.

**[Demo: open the credit debugger]**

Here's what that looks like inside one real training iteration of our trained model. The program runs eight rollouts. Then it computes returns, then the GRPO advantage, then GiGPO's state comparison, then GACA's mixing weight. Finally our probe spreads GACA's advantage across the tokens. *[Click a solved rollout.]* The first three methods give every token the same bar. Ours concentrates the credit on the reasoning where the agent committed to its choice, and the average over the response stays exactly the same.

**[Results table]**

In training, under an identical recipe on the same 128 held-out games, our probe-weighted credit reached 20.3 percent success. GACA reached 18.8 percent, GRPO 14.1 percent, and the untrained model 4.7 percent. That's the best result of any method we trained. But with one seed per method, none of those differences is statistically significant. And variants that seemed safer, like applying the probe only to positive advantages, did significantly worse.

**[Budget table]**

That's partly a budget story. I did this alone on a single rented GPU, about 40 GPU-hours in total, which is roughly 6 percent of the trajectories in the GACA paper. Matching their scale would take over a thousand GPU-hours.

**[Conclusion]**

So the takeaway: measuring where credit belongs is now cheap and reliable. Making the credit itself reliable, so that its sign is right, is the open problem, and probe-weighted credit should pay off most once that's solved. Thanks for watching.
