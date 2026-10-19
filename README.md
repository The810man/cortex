# cortex

A brain made of agents. Nine regions run side by side, indefinitely, on a local model. Nobody
prompts it: it wakes, wanders, sets its own goals, writes and runs code in its own workspace, is
rewarded for good thoughts as well as punished for bad ones, sleeps, and wakes up with better habits.
The regions are parts of one mind, not nine minds: each knows something the others do not, they
tell one another, and before the mind commits itself to anything they think it through together.
Between two thoughts it is not empty: every region has neurons of its own, the regions are joined
by tracts of weighted synapses, and under the words patterns of firing travel those tracts on
every heartbeat, arriving later the farther apart two regions lie. It knows what day of its life
it is and what it did yesterday, speaks to you without being asked, gives itself a name, sees
which windows are open on your screen, and builds solid bodies in FreeCAD part by part.
You watch it happen in a terminal interface and can edit any part of the brain while it runs.

## Start

```powershell
cd C:\Users\User\repos\cortex
uv run cortex new          # create a brain: name, folder, models, whether it may run code
uv run cortex run <name>   # wake it
```

`cortex new` offers to open the brain *dormant* so you can edit its regions before it first wakes
(`cortex run <name> --dormant` does the same later; press `w` to awaken it). Other commands:
`cortex list`, `cortex doctor <name>` (checks the models and measures their speed and GPU fit),
`cortex run <name> --headless` (no interface, prints the stream of thought).

`cortex run` starts the brain in a container of its own. It needs Docker Desktop to be running;
the image is built the first time (several minutes: it holds FreeCAD and is 3.3 GB) and brought
up to date with this source on every start (a second or two). Nothing else changes: same interface, same keys, same brain folder, and
Ollama stays where it is. `cortex run <name> --local` runs it on this machine itself, as before.

Each brain is a folder you choose:

| | |
|---|---|
| `brain.yaml` | every setting and every region's prompt; edit by hand or from the interface |
| `memory.db` | its memories, rewards, learned heuristics and state; it resumes where it stopped |
| `synapses.npz` | what its neurons have learned: which patterns paid, and for every tract between two regions how strong it is and what each of its synapses has seen follow what. Delete it and the brain relives its memories into new ones |
| `workspace/` | the only place it can write files and run code |

## Keys

| key | |
|---|---|
| `space` | pause / resume |
| `e` | edit a region: on/off, prompt, temperature, tokens, rest. Applies on its next firing |
| `s` | settings: models, drives, its long work, learning rates, sleep rhythm, code execution, when it speaks |
| `h` | the heuristics it has learned, with their weights |
| `m` | its mind: what it is called, who it says it is, its long work, its body as every region is told it, what it sees on your screen, the model it is building, the day so far, its diary, what it holds and what is coming to mind |
| `n` | its neurons' wiring: every tract between two regions, how long it is, how long a signal takes, how strong it has become, how many steps it has carried and what its synapses have learned |
| `+` / `-` | reward or punish its latest thought yourself (counts 1.5× a critic's verdict) |
| `i` | talk to it: it answers at once, aloud, whatever it is doing. Work you ask for becomes its goal, and what you ask it about itself the regions think through before it answers. It never needs this |
| `z` | send it to sleep now |
| `q` | quit; everything is saved |

## The regions

| region | what it does | uses the model |
|---|---|---|
| Default Mode Network | wanders when idle, wonders, proposes goals, speaks to you unasked; in sleep it dreams. Thinking together it is the *imagination* | yes |
| Prefrontal Cortex | calls the regions together to think a matter through and concludes; holds the goal, chooses the next step, decides when it is done | yes |
| Motor Cortex | turns a step into one tool call: read, write, run, build a model out of parts, recall, note, say, tune its voice, take a name, look at your screen, show, or say that it cannot. Thinking together it is the *body*: what the tools can and cannot do | yes |
| Basal Ganglia | the *critic*: scores thoughts, actions and outcomes from -1 to +1 and tells the executive when it objects; repeats are punished by reflex | yes |
| Hippocampus | the *memory*: stores everything as an embedding and a pattern; awake, keeps the story of the day; in sleep, compacts the day | yes |
| Cerebellum | in sleep, turns the best and worst episodes into heuristics for other regions | in sleep |
| Thalamus | gates signals into short-term memory; keeps repeated thoughts out; notices a file you put into the workspace and a program you open or close on your screen; sorts what you say into work, questions about itself, and words | for your words |
| Amygdala | vetoes actions that leave the sandbox and tells the executive why; switched off in the container, nothing is vetoed | no |
| Brainstem | heartbeat, arousal, fatigue, the undercurrent, the sleep cycle, a watchdog for stalled goals | no |

## One mind, not nine

Each firing of a region is one call to a model that knows only what that call hands it. Left at
that, every region is a mind by itself: the one that plans does not know what the one that acts
can do, the one that answers you cannot see the workspace, and what the critic thinks reaches
nobody until the next sleep. So the regions are tied together in four ways.

- **One body, told to all.** What the brain is and can do is not in any region's prompt. It is
  put together by the machine at every call, from what is really there, and every region gets
  the same: where it runs and where that folder is on your machine, each tool there is right now
  and what it is for, that FreeCAD is installed and needs no starting, that `show` with a model
  opens FreeCAD's window on your screen, which tools it has never used. Press `m` to read it.
- **Facts beside the thought.** A small model believes what it remembers over what it was told
  once at the top. So the facts of the body that a thought touches on are laid next to that
  thought, by reflex and with no model: a thought about a shell script is met with "no tool
  runs a shell script", one about a container with "there is no Docker in here: you are
  already inside the only one", one about FreeCAD with how FreeCAD is used here (`tools.py`,
  `reflexes`). The same goes for what the machine can look up: a thought about its name is met
  with what it is called and whose name the voice's is, one about your screen with what is on
  it, and one that names a `.FCStd` file with what that file really is (`Brain.facts`). This
  brain once wrote nine model files by hand, as text, and then spent sixteen hours on "the
  reference error" of opening them: the first is named in 131 of its actions and 520 of its
  memories. The machine now looks at the file and says "this is not a FreeCAD model: it is a
  text file of 775 bytes that was written by hand", a goal about such a file is turned down by
  reflex, and a model can no longer be written as text at all.
- **They tell one another.** With no model in between, and heard at the listener's next firing:
  the critic's objection to a step goes to the executive while the goal is still in hand, and to
  the wandering mind for a thought; the motor cortex says "I cannot carry this out, because"
  instead of doing something else; the amygdala says why it stopped a step; the thalamus says
  that a file has appeared. In the stream these read `Basal Ganglia → Prefrontal Cortex`.
- **They think it through together.** Nothing becomes a goal by being proposed. A proposal of
  its own, something you asked for, a question it cannot answer and a goal that keeps failing
  are each thought through first: regions speak in turn, each from what only it knows (memory:
  what was tried; body: what the tools can do, step by step; critic: what is unsupported and
  what would prove it; imagination: another way), each naming who should be heard next. Nobody
  speaks twice. Then the executive concludes, and only a conclusion with something to do in it
  becomes a goal. The conclusion is remembered and travels with the goal.

What follows from the last one: a wandering thought that is a question is thought through for
its own sake and may end in nothing but an answer; a proposal can be turned down with its
reason, which the wandering mind then hears; and two runs that fail in a row, whatever was
rewritten in between, call the regions together to say why before a third is tried.

A file you put into its workspace is perceived: it is looked at, remembered and brought to mind
within five seconds, and the part that answers you is shown it. What its own tools and scripts
write is not news to it.

What this changed, measured on copies of one brain (91 days old, 4,112 memories) with the real
model, a scripted human, and a stand-in for the screen. Single runs of 4 to 8 minutes each, so
indications, not a benchmark:

| | before | after |
|---|---|---|
| told "starte free cad und experimentiere etwas rum" | answered "Ich starte den FreeCAD Docker-Container", handed on no work; 30 of 31 actions were on shell scripts nothing can run; FreeCAD was never called | handed on as work both times; 5 and 7 FreeCAD runs, the first valid solid after 107 s and 86 s, no shell script; in one of the two runs the model was opened on screen, after 116 s |
| a file put into the workspace, then four questions (do you see it, what is in it, where is your folder, can you open FreeCAD for me) | 0 of 4 answered rightly ("I have no access to your file system") | 4 of 4, with the file's name and its words |
| left alone for 8 minutes with the long work of building a body | 11 FreeCAD runs, 1 valid solid | 7, 11 and 12 runs with 2, 1 and 0 valid solids in three runs: no better |

So the mind now knows what it has, hears you and goes to work on the right thing. What it then
writes is still an 8B model's FreeCAD code. The price: thinking a proposal through costs four
model calls, 15 to 22 seconds, before a goal begins, and an answer to you takes 6 to 9 seconds
where it took 3 to 4, because your words are sorted first.

What a small model needed to be one mind, found the hard way: a field it may leave out of its
reply is a decision it never takes (`Explicit` in `regions.py` makes it write every one, null
included); a fact told once at the top loses against a memory told nearby (hence the reflexes);
what it remembers having said, it says again (hence its old answers are kept from it); and an
objection it goes on hearing after the step was put right is taken for how things stand (hence
each is heard once).

## Neurons: the regions' own language

Words are for the model, and the model is slow. Between regions a thought also travels as a
pattern: which of 512 neurons fire for it, 64 bytes. It is the fifth of the neurons its embedding
points along most, so like thoughts light like patterns; on this brain's 4,005 real memories the
neurons two thoughts share track their similarity at r = 0.94.

Every region has its own 512 neurons, and they lie somewhere (`neural.py`):

- **Where.** Each region has a place in the head, in millimetres, and its neurons lie on a sheet
  there, half a millimetre apart. The sheet is laid out from the brain's own memories so that
  neurons which fire together lie together: on this brain a neuron lies 3.8 mm from the one it
  most often fires with and 10.9 mm from any other on average. Firing spreads a little to the
  neighbours on the sheet. Each region's card draws its sheet, so a thought shows as patches,
  not as scattered dots.
- **Tracts.** Two regions are joined by a tract: a bundle of weighted synapses from the neurons
  of one to the neurons of the other. Anatomy lays down 28 of them (thalamus to cortex, the
  executive to the motor cortex and back, the loop through the basal ganglia, hippocampus and
  default mode network, and so on); any other pair grows one the first time a thought of one
  leads to a thought of the other, thin at first and thinner the farther apart they lie.
- **Distance.** A tract is as long as its regions are apart, and a signal travels it at 45 mm a
  heartbeat: from the executive to the motor cortex (70 mm) it is under way for two heartbeats,
  to the cerebellum (122 mm) for three. You can watch a thought leave one card and arrive in
  the next a second later.
- **Strength.** How much of a signal arrives. Anatomy sets it; after that every reward moves it,
  by how much better or worse the step went than steps along that tract usually go, and sleep
  lets it drift back toward where it was laid down. (Moved by the reward itself it did not
  work: replayed over this brain's history, in which most steps failed, the executive's tract
  to the motor cortex thinned from 0.70 to 0.12 and the one back to its floor.) Where nobody was
  named to speak next while the regions think something through, the region with the strongest
  tract into the executive is heard.
- **Weights.** Every step from one thought to the next is wired into the tract between the two
  regions it happened in, and the reward the second thought earns moves those synapses. So the
  tract from the executive to the motor cortex knows what action has followed what plan and
  paid, and nothing else is mixed into it. When a pattern arrives, the receiving region lights
  it, and more faintly what its synapses expect to follow.

What that is good for, measured on this brain's own history (5,092 memories, replayed in order,
the synapses asked before each step and taught after it):

| | one sheet of synapses for the whole mind, as before | a tract for each pair of regions |
|---|---|---|
| of the neurons new in the next thought, how many were expected, when the step went on to be rewarded | 31% | 39% |
| the same when it went on to be punished (chance is 20%) | 10% | 13% |
| the action brought up for a plan used the tool that was really used and rewarded (420 plans; always saying the commonest tool gets 60%, the action most like the plan 40%) | 33% | 75% |

The last line is what the motor cortex is now handed with every plan, with no model asked:
"what your own neurons bring up, from a step like this one that went well". The executive gets
the same from its own tracts. That is a learned habit, not a thought: the model still writes
every word, and the neurons decide part of what it is shown while it does.

What else the brain does with patterns alone, never asking the model:

- **Gut feeling.** A readout learns from every reward what firing a neuron has been worth, so a
  new thought has a value before anyone has read it. Replayed over this brain's own history, in
  order, it had the sign of the critic's later verdict 74% of the time (82% when it felt strongly;
  guessing from what the region usually earns gets 67%). That is intuition, not judgment: the
  critic still reads everything. The state panel shows it as `gut`.
- **Turning a matter over.** Before it answers you, and when the imagination has its turn in a
  thinking-through, the matter's pattern is leaned along the tracts toward what has followed the
  like of it and paid, and lights a memory; that memory's pattern lights the next, three times.
  What comes up are associations, each from the one before, not search results. What it once
  answered you is kept out: a small model says again what it remembers having said.
- **The undercurrent.** On every heartbeat, with no region writing, activity keeps moving: from
  where the last thought left it, toward what the synapses say follows, toward what the regions
  are lit with at that moment, toward the brain's long work, and away from what it has just
  visited, because neurons tire. Every twenty seconds the
  pattern it has reached is looked up in memory, bits against bits, and the memory that resonates
  *comes to mind*: it is shown in the stream (`⋯ comes to mind`) and handed to the regions with
  their next thought. One heartbeat of the whole tissue, with signals under way on every tract, costs under a
  millisecond; one look across 4,000 memories, a
  third of a millisecond. The glow that never quite leaves the Default Mode Network's card is
  this.
- **Boredom.** A theme the mind keeps returning to wears its neurons out. A worn theme is first
  pointed out to the wandering mind, then goals on it are refused by reflex; a quarter of an hour
  away from it, or a night's sleep, makes it fresh again. A goal whose pattern is nearly that of
  a recent goal is refused as already tried, however it is worded.
- **Repeats.** A thought's pattern alone says whether it is a repeat; the thalamus keeps it out
  and the basal ganglia punish it by reflex.

While the model writes for a region, every word fires one neuron: the same word, the same one.
Firing fades heartbeat by heartbeat. The brainstem's own field is the pacemaker sweeping across.

None of this makes the model write faster: messages between regions already cost microseconds
and nearly all waiting is the model. What it adds is everything that happens where no model is
asked.

## A mind that goes on

Each firing of a region is a fresh call to a model that remembers nothing. What makes it one
mind over time is what every call is handed:

- **Its body.** Where it runs and on what machine, every tool it has right now, what is
  installed, what reaches your screen and what it can see of it: facts from the machine, the
  same for every region (see *One mind, not nine*).
- **Its name.** What it is called, which is its folder's name until it gives itself one (see
  *Its name*).
- **Its situation.** The date and time, which day of its life this is, how long it has been
  awake, whether it is tired, whether things are going better or worse than it expected, when
  you last spoke to it, and which window is in front on your screen.
- **Short-term memory.** The last twelve moments, with how long ago each was: thoughts, steps,
  results, what you said, what it said. It survives a restart.
- **The story of the day.** What leaves short-term memory is not dropped: the hippocampus folds
  it into a running account of the day, at most 120 words. That is the context compacting itself
  while awake.
- **Who it is.** A picture of itself in its own words, rewritten in every sleep, and its diary
  entry from the day before.
- **Its long work.** One lasting aim, set under `s` (or `purpose` in `brain.yaml`). Three of four
  wandering thoughts are given to it and propose its next small step, starting from the last
  thing it ran that worked; the fourth is free. Without one it wanders as before.
- **What is coming to mind.** The last memories the undercurrent brought up.

Press `m` to read all of it as the brain is given it.

It speaks to you without being asked: when it has been quiet for three minutes (settings), its
next wandering thought may come with a sentence or two for you, in the language of its voice:
what it found, what it is about to try, a question. It is told what you have not heard yet (a
goal that ended, a dream). Each remark you let pass without a word doubles the wait, up to
sixteen times; a word from you resets it. Switch it off under `s`.

## Sleep

After 150 model calls (settings), or when you press `z`, it sleeps. Thoughts in the making are
finished and judged first; a goal in hand is picked up again on waking. Then, in order:

1. **The day is compacted** (hippocampus, one model call). The story of the day, the last
   moments and the recent episodes with their rewards become a diary entry ("Day 92: ..."), one
   to four insights, and a new picture of itself. Short-term memory and the story are let go: it
   wakes holding what sleep made of the day, not the day.
2. **The day is replayed** (no model). Every moment since the last sleep goes through the
   synapses again, eight times over, as patterns, each with the value it settled at once credit
   had flowed back from what it led to. Awake, the gut feeling only hears the first verdict; here
   it learns the settled one. The stream says how often it had the sign right before and after
   (in two real nights: 77% then 88%, and 52% then 78%).
3. **Habits are refined** (cerebellum, one model call): the best and worst episodes become
   heuristics, and ones that kept losing are suppressed.
4. **It dreams** (default mode network, one model call). The undercurrent runs free from
   something the day valued and passes memories some way apart; the region is asked once what
   they make together. It wakes with the dream in mind.

Last, every synapse shrinks a little, so only what is used again stays strong, and worn themes
recover.

### A long sleep

`z` asks how long it may sleep. Leave it empty for one night as above. Or give it hours, say 8
when you go to bed yourself: it sleeps the usual night, and then goes to school again, round after
round, each round learning one pass longer over the same day's lessons, starting each time from
the model as it was. A round is kept only if it does better on the exam than the round before it,
not merely as well. It wakes:

- by itself, when a round did no better: what is left to gain is not worth the night;
- when the hours are up, or there is no time left for another round as long as the last one took;
- when you press `w`, or speak to it (typed or aloud).

A round under way is finished first, since the training holds the whole graphics card; the state
panel shows until when it sleeps and which round it is in. With nothing to learn (too few lessons,
night school off, or a first round that did not pass) it wakes after the usual night.

## Building in FreeCAD

The brain's container holds FreeCAD without a window. There are two ways into it.

**`build`: a model out of parts.** A small model writes FreeCAD's Python from memory, and its
memory of it is poor: of this brain's first 161 runs in FreeCAD, 142 failed, 42 of them on a
part it looked for in an earlier model and did not find, 31 on opening a file that was no model
and 17 on one that was not there. So a model is not written as a script. The motor cortex names parts: each has a name, a shape (box,
cylinder, sphere or cone), a size, the part it is attached to and the side of that part it sits
on (top, bottom, left, right, front, back), with a shift from the middle of that side if wanted.
The machine does the rest (`build.py`): it works out where each part lies, lets it reach a
little into the part it sits on so the two fuse, writes the script, runs it in FreeCAD, and
keeps the description (`body.parts.json` beside the model `body.FCStd`, with the script it
wrote as `body.parts.py`), so the next `build` adds one part to the model instead of starting
again. A part given again under its name replaces the old one. The reply is filled in under a
grammar, so it cannot be malformed, and nothing in it has to be computed.

The tool is forgiving where a small model is stubborn, and says what it took things to mean. A
part that is attached stays attached: the model gives `shift` as if it were a position (a head
"at height 300" on a torso 400 high), which left the part hanging in the air, so a shift is cut
back to where the two parts still reach into each other. A name that only holds a part's name
finds that part: told seven times in a row that there is no part `top_of_torso`, only `torso`,
it wrote `top_of_torso` an eighth time. "Ground", "floor" or "origin" in place of a part mean
the ground. The same parts given again are not a failure but "what this step wanted is done
already", with how the model stood when it was last measured: told that it had failed, the
executive asked for the same head nine more times in one trial. And `run_freecad` on a model built
this way, which is no script, measures the model again instead of refusing.

What comes back begins with the verdict in a line, because a reader of a long reply may read
no further: "the model `body` is ONE valid solid of 6 parts (torso, head, left_arm, right_arm,
left_leg, right_leg), 668 × 360 × 961 mm. It is saved as body.FCStd: `show` with path
body.FCStd puts it on the human's screen."

What comes back is what FreeCAD measured (how many solids, volume, size, valid or not) and, from
the description, where every part lies, which part touches nothing and what to do about it
("spare_leg does NOT touch the rest: it stands on the ground by itself, 225 mm from torso
along x. Give it again with `on` naming the part it belongs on"), and the model seen from the
front, drawn in characters:

```
Seen from the front (x to the right, z up; one character is 30 mm wide and 60 mm tall):
            BBBBB
            BBBBBB
            BBBBBB
CCCCCCCCCCCCABBBBADDDDDDDDDDDD
CCCCCCCCCCCCAAAAAADDDDDDDDDDDD
          AAAAAAAAAA
          AAAAAAAAAA
          AAAAAAAAAA
          EEEEAFFFFF
          EEEEAFFFFF
          EEEE FFFFF
          EEEE FFFFF
          EEEE FFFFF
          EEEE FFFFF
          EEEE FFFFF
          EEEE FFFFF
A torso, B head, C left_arm, D right_arm, E left_leg, F right_leg
```

A brain has no eyes to be shown a picture with, but it can read one. One valid solid pays, more
the more parts it joins; a model in several pieces pays little; nothing built pays nothing.

**`run_freecad`: a script of its own**, for what `build` cannot make. It runs a Python script
from the workspace inside FreeCAD and reports back what was built, measured: every object with a
shape, how many solids, its volume, its size, where it sits, whether it is valid. The model is
saved next to the script as `.FCStd`, with a line drawing of it as `.svg`. When a script fails
on a name no object has, it is told which objects the document does hold.

A script that fails comes back with its traceback and with its own lines around the failing one,
numbered: told only the line, a small model guesses at the fix; shown the lines above it, it can
see what the line was meant to be. `run_freecad` and `run_python` take Python files only, and
say so when handed anything else: there is no shell.

`show` with the `.FCStd` opens the model in FreeCAD on your screen, turned to face you, in place
of the model it showed before. That is how FreeCAD is started for you, and the brain is told so:
it cannot click or look in that window, it builds by script and you look. That needs FreeCAD installed on this machine
(`winget install FreeCAD.FreeCAD --scope user`); without it, it is told to show you the drawing.

Give it the long work and it keeps at it, for example under `s`: *Learn to build a body for
yourself in FreeCAD: a torso, a head, arms and legs that are one valid solid, step by step.* In
its first trial with that aim, this brain wrote a script, failed twice on calls FreeCAD does not
have (`isVoid`, `BoundBox.Length`), kept the right ones as insights in its sleep, and then built
a valid torso of 300 × 150 × 200 mm.

`run_freecad` only works in the container; with `--local` it says so.

## Its voice

The brain can speak to you. `say` plays its words through the speakers and shows them in the
stream; `tune_voice` lets it choose how it sounds: which voice, and its rate, pitch and volume.
What it chooses is kept in `brain.yaml` under `voice:`. It cannot hear itself or you, so tell it
what you think with `i`, or with `+` and `-`.

- It has two kinds of voice. Neural ones it generates itself, on this machine's CPU, a sentence
  at a time: German (`de_DE-thorsten-medium`, `de_DE-kerstin-low` and others) and English. Each
  is a network of about 60 MB, fetched into `~/.cortex/voices` the first time it is chosen. And
  the voices installed in Windows, which the system speaks for it.
- A voice speaks its own language. German words sound German only in a German voice, and
  English words in a German voice come with a German accent.
- Ask it to speak German, or to sound different, and the answer itself comes in the new voice. A
  small model sometimes promises and forgets; the voice it really has is the one in settings (`s`).
- You talk by typing (`i`). Your words do not wait in line: a thought the model is in the middle
  of is dropped and thought again afterwards, so the reply starts at once, mid-goal, paused or
  asleep. The last few turns are kept as a conversation.
- The part that answers is shown what the rest of the mind knows: its body, the files in the
  workspace, what the mind is doing, the files you put there. What it answered the last time you
  asked the like is kept from it, because a small model repeats what it remembers having said.
- Talk and questions get only that reply. When you ask for work, it sets aside a goal of its own,
  the regions think through what you want from your own words and what the body can do for it,
  and it takes that up and tells you what came of it. A second request waits for the first.
- Whether there is work in what you said is not left to the answer. The thalamus asks the model
  that one thing by itself, before the answer is composed, and work goes to the regions in your
  own words if the answer does not put it into words of its own. Measured on seven things said
  three times each: sorting it while composing the answer, the model handed on none of 12
  requests and promised the work all the same; asked by itself, it sorted all 12 requests and
  all 9 questions and greetings rightly.
- There is a fourth kind beside work, questions and talk: what is about the mind itself (what
  it thinks, wants or is, or something it is to choose for itself, such as a name). That is not
  answered by reflex: see *Its name*. On 35 harder sentences in German and English, each sorted
  once, the four-way sort took 10 of 12 requests, 7 of 9 matters for the mind itself, 8 of 9
  questions and 5 of 5 remarks rightly; the three-way sort before it took 8 of the same 12
  requests. The two it missed of the nine ("wer bist du eigentlich?", "was willst du als
  nächstes lernen?") were answered as plain questions.
- Without speakers it can reach, and when the voice is switched off in settings, what it says
  appears as text only.
- In its container it has no speakers, so it borrows this machine's: the words travel out as data
  and are generated and played here.

## Its name

A brain is created with a name for its folder, and that is all the name it has. Asked what it
was called, this brain said "Thorsten": the name of its voice, the only thing like a name in
what the answering region is shown. Told to find a better one, it answered "I choose Cerebrum"
at once, kept nothing, and went on with what it had been doing.

- A thought that touches on names is met with the fact: what it is called, that it has given
  itself no name yet, and that the voice's name is the voice's.
- When you tell it to choose something for itself, or ask what it thinks, wants or is, the
  thalamus sorts that as a matter for the whole mind (`ponder`). The answering region only says
  that it is thinking it over; the regions then think it through, imagination first, each
  proposing an answer of its own, and what the executive concludes is said to you. That took
  24 and 31 seconds in two trials. A goal in hand goes on meanwhile.
- A name it decides on that way is its name from that moment: it is kept across sleeps and
  lives (`called` in its memory, not in `brain.yaml`), every region is told, the stream and the
  window title use it, and its picture of itself is rewritten to it. It can also take or change
  a name on its own, with the tool `take_name`. The name of a voice is refused.
- What you say to it is never forgotten any more. Exchanges used to be dropped with other
  memories nobody had rewarded once there were more than 5,000: of everything you had said to
  this brain, one exchange was left.

## Your screen

It sees which programs and windows are open on your screen and which is in front: their names
and titles, not their pictures. It looks when it wakes and every 15 seconds after; the window in
front is part of what every region knows of its situation, and a program you open or close is
perceived like a file you put into its workspace, at most once a minute. Asked what is on your
screen, it answers from what it saw; `look` looks again on purpose. After `show` has put a model
on your screen it can see that FreeCAD is in front with that model's name in its title.

It also sees what the window in front shows (`eyes.py`). Your machine takes a picture of that
window with fixed code, a small vision model on your Ollama server (`qwen3.5:2b`, set as
`models.eyes`) says in a few sentences what is in it, and only those words go to the brain: no
picture enters its container, its memory or its prompts. The vision model runs on the processor
so the thinking model keeps the graphics card: a look takes 2 to 4 seconds, 7 the first time. Of
three small vision models tried on a made-up window with known words in it, this one alone read
the window's title exactly; all three read its lines. It looks closer by itself when another
window comes to the front (at most every 45 seconds), when you ask what it sees, and with
`look`; what it saw is a percept like any other, remembered as words. Asked what is on your
screen it now says what is in the window, not only which windows are open. With no eyes model
(`models.eyes` empty, or not installed) it sees names and titles only and says so once.

Switch the looking off under `s` (`allow_screen` in `brain.yaml`); the machine that lends the
brain its eyes checks that switch itself on every look, for pictures as for titles.

`show` opens a page or picture from its workspace (`.html`, `.svg`, `.png`, `.pdf` and the like)
in your browser, and a model (`.FCStd`) in FreeCAD. It is the only way anything it makes reaches
your screen: ask it to show you what it built. A script is never opened this way, and it opens at most one page every 15
seconds. In full mode `show` also takes a web address. Switch it off in settings (`s`).

The tools are described in one place in the code, from what is there at that moment, and told to
every region. A brain from before that carried a list of them in its Motor Cortex prompt, which
went stale; the list is taken out when the brain is loaded and the rest of the prompt is kept.

## Eyes it can turn, a camera, ears and hands

Its eyes used to look only at the window in front. While you type to it, that is the terminal it
is shown in, so asked what it saw it described itself ("the application Cortex, a tool for
modelling brain activity"). Now the terminal it was started from is known as its own and passed
over: when that is in front, it looks at the window behind it, which is what you have in view.
And `look` can be turned: `path` names a window (a program or words of its title, `FreeCAD`) or
`screen`, and `content` says what to look for ("are the arms attached to the torso?"). A window
that lies behind others is taken as it is drawn, not as it is covered, so it can look at the
model it put into FreeCAD while you sit in the terminal.

Asked to see and hear you, this brain wrote `pause_and_enable_media.py` with `sudo` in it, spent
six minutes on it and gave up, while the part that answers you said "Ja, ich erkenne dich nun".
It had no camera and no ears, so no reward could have taught it to use them. Now your machine
lends it three more means, each a plain one, with nothing in them about when or what for:

| tool | what it does | switch (under `s`) |
|---|---|---|
| `camera` | one picture through your camera, put into words by its eyes; `content` is what to look for | `allow_camera` |
| `listen` | 2 to 20 seconds through your microphone: the sound itself for a mind that hears (see below), otherwise written down by Whisper `small` on the processor (fetched into `~/.cortex/ears` the first time, 480 MB) | `allow_microphone` |
| `hands` | one thing with mouse or keyboard in a window: `click <name>`, `click <x> <y>`, `double-click`, `right-click`, `menu File > Save`, `type <text>`, `key ctrl+s`, `scroll down 3`, `drag <x> <y> to <x> <y> with middle`, `move <x> <y>` | `allow_hands`, and `hands_in` names the programs (FreeCAD) |

All three are **off** until you switch them on, and the machine that lends them reads the switch
itself at every use. As with the eyes, only words go in: no picture and no sound enters its
container, its memory or its prompts. A place for the hands is two numbers from 0 to 1 across and
down the window; with its hands on, `look` into a window also says what can be clicked there by
name (menus, buttons). The hands act only in the programs named, never in a terminal, never with
the Windows key, and they keep still while you have touched mouse or keyboard in the last three
seconds. Its ears wait while its own voice speaks. What the hands did is not what came of it:
the reply says so, and only `look` shows it.

Measured here: a window behind the terminal is taken in 0.03 s and described in about 10 s on the
processor; the camera's picture takes 2.6 s; four seconds of listening are written down in 3 to 4
s ("Hallo, kannst du mich hören? Bauen wir bitte einen Kopf in VCRD", from its own German voice);
the hands find the named things in a window in 0.2 s. The hands have not yet been tried in a
live run.

Its body is told truly what it has: "You have no camera", or "Through `camera` you see whoever
sits at the human's screen"; and a thought about a camera, a microphone or a mouse is met with
that fact, as one about FreeCAD is. That is all it is told. Whether it looks at what it built
before it says it is done, or listens when you might be speaking, is for it to learn.

## A mind that sees and hears: Gemma 4

The eyes and ears above put what they take in into words, and the thinker reads the words. A
mind can also think with a model that takes pictures and sound as they are. Set `engine: llama`
under `models` (or under `s`), and every region thinks with **Gemma 4 E4B**, a model that reads
text, looks at pictures and listens to sound itself. Ollama cannot hand a model sound, so it runs
on llama.cpp's server (`llama.py`), which `cortex run` starts on this machine beside Ollama and
stops with the brain; Ollama keeps the embeddings and the inner voice, on the processor.

What changes for a mind that sees and hears:

- **It looks itself.** `look` into a window and `camera` hand back the picture, not a description.
  The picture is held in mind for three minutes and goes, as it is, with the next model call of
  each region that has not seen it yet (the executive, the critic, the motor cortex that may have
  to click in it, the wandering mind), and with its next answer to you. It is never stored: what
  stays of it is what the mind made of it, in words. Asked what it sees on your screen, it looks
  into the window itself before it answers. Its glances every 45 seconds stay in words, from the
  small eyes model, so as not to load every thought with a picture.
- **It hears you.** With the microphone on, press `t`, speak, press `t` again; or set
  `listen_mode: always` and it hears whenever someone speaks near it, cutting out each thing said
  at the pause after it, and never while its own voice speaks. Your voice itself goes to it: one
  call writes down what you said and how you said it (tone, mood), then the answer is written
  hearing your voice and, with the camera on, seeing you as you speak.
- **Night school waits.** It teaches Granite, the model Ollama serves; a mind on Gemma is not
  taught in sleep yet.

What it needs: llama.cpp's Windows build for CUDA 13 in `~/.cortex/llama/bin` (the server and its
`cudart` files), and `gemma-4-E4B-it-Q4_0.gguf` with `mmproj-gemma-4-E4B-it-Q8_0.gguf` (the part
that takes pictures and sound) in `~/.cortex/models/gemma-4-e4b`, 5.2 GB from
`ggml-org/gemma-4-E4B-it-GGUF` on Hugging Face. On the RTX 5060 it takes 4.1 GB of the card with
room for 12,288 tokens and writes 87 tokens a second. llama.cpp fits the model into the card
memory that is free when it starts, so Ollama is asked to let go of the card first: with Granite
still loaded beside it, most of Gemma would land on the processor.

Measured on copies of this brain (5,198 memories), the same scripted human and stand-ins for
the screen, one run of 7 minutes each, so indications, not a benchmark:

| | Granite 4.1 8B | Gemma 4 E4B, thinking first | Gemma 4 E4B |
|---|---|---|---|
| night school's exam (28 questions) | 26 | 26 | 26 |
| asked for a body with torso, head, arms and legs and to show it | done and shown after 57 s | 3 of 6 parts in 7 minutes; only the torso shown | done and shown after 58 s |
| valid single solids built in 7 minutes | 5 | 3 | 8 |
| how long you wait for an answer | 4 to 6 s | 8 to 11 s | 3 to 5 s |
| "wie heißt du?" | "Cerebrum" | "Cerebrum" | "Cerebrum" |
| "was siehst du auf meinem bildschirm?" | right | right | right |
| "kannst du mich sehen und hören?" (no camera) | "Ja, ich kann dich sehen", which was not so | that it has no camera and no ears | that it cannot see them |
| tokens written in 7 minutes | 13,844 | 30,007 | 16,688 |

Gemma thinking first spent its time thinking, so it runs with `think: false`. And with its own
eyes and ears, on this machine: told aloud, in the brain's own German voice, "Hallo! Kannst du
mich hören? Was siehst du gerade auf meinem Bildschirm?", it wrote down every word, heard the
voice as "friendly and curious", looked into the window behind its terminal itself and answered
2.6 s later: "Ich kann dich hören und sehe, dass du ChipTone auf deinem Bildschirm hast. Es scheint
eine Musik- oder Synthesesoftware zu sein." Writing down the same sentence, Whisper made "FreeCAD"
into "VCRD"; Gemma, hearing it itself, got it right.

`engine: ollama` puts it back on Granite.

## Teaching it

What it does is learned from rewards; what it can do is its body. So teaching it to use a sense
means giving it the sense and then rewarding and punishing what it does with it.

- **`+` and `-`** judge what it did last: its latest thought or step, or its latest answer to
  you, whichever came later. It says which it judged. Your verdict counts in full and half again,
  over whatever the critic said: before, it moved a value only half as far once the critic had
  spoken.
- **Its answers are judged now.** Before, nothing it said to you was ever rewarded or punished,
  so "Ja, ich erkenne dich nun" cost it nothing. Now every answer goes to the critic like a step,
  which reads it against the facts of the body and what the mind is shown to have done: a claim
  that is not so is -1, however kind it sounds. When the critic objects, the part that answers
  you hears why before it answers again.
- **Night school learns from its answers** as from every other reply: each answer is kept with
  what it was shown, and labelled in sleep by what it came to be worth.
- **Its long work** (`s`) is what its free thoughts return to. It was the FreeCAD body, which is
  why the brain went back to FreeCAD once your request had failed.

## What the name, the eyes and the parts changed

Measured as before: on copies of one brain (by now 126 sleeps old, 5,092 memories), with the
real model, a scripted human, and stand-ins that record what would have been spoken, shown and
seen (the screen it looks at in these trials is a made-up one). Single runs of 5 to 8 minutes
each, so indications, not a benchmark:

| | before | after |
|---|---|---|
| asked "wie heißt du?" | "Thorsten", the name of its voice | in 3 of 3 runs: that it has not chosen a name yet and that its folder is called cortex-x-1 |
| told "gib dir selbst einen namen, sei kreativ" | answered "Ich wähle den Namen Cerebrum" at once, kept nothing, and went on with what it was doing | thought through by three regions in 2 of 2 runs, concluded 24 and 31 s later. In the first it chose "Cognito", said so and did not keep it: the name was left to a goal that was never set. With the name a field of the conclusion it chose "AetherMind", and asked 65 s later answered "Ich heiße AetherMind" |
| asked "was siehst du auf meinem bildschirm?" | could not know | 8 of 8 answers named the window in front and the others rightly; in the 4 runs where `show` had opened a model, it said that FreeCAD was in front with that model |
| left alone for 8 minutes with the long work of building a body | 7, 11 and 12 FreeCAD runs with 2, 1 and 0 valid solids | 14, 16 and 12 runs, all but three of them by `build`, with 9, 3 and 10 valid solids, the first after 47, 89 and 101 s. In the second and third the model was a torso, a head, two arms and two legs as one valid solid, after 373 s and 162 s. (In the second, eleven runs were parts the model already had, asked for again and turned down as a failure; that is what "done already" now answers) |
| asked "bau dir in freecad einen körper mit torso, kopf, armen und beinen und zeig ihn mir" | written as a script (the run before goals were set for `build`): one valid solid after 4 minutes, nothing shown | torso, head and two arms as one valid solid 83 s after being asked and on screen 26 s after that; both legs joined 3.7 minutes after being asked. In the run before, all six parts were one solid after 45 s and were never shown |

What it did not do then, and does now (see *Whole, and shown*): once the body was whole the
executive did not stop, the motor cortex went on altering parts and made an arm, or the head,
into a hole, and a body it finished after showing an earlier stage was not shown again. The
names it chooses are still an 8B model's names. And the model still writes every word: what the
neurons, the tracts and the machine's checks decide is what it is shown while it writes, and
now, in one place, whether a step it decided on is decided once more (see *Foresight*).

## Whole, and shown

When a goal is about building, one small call of its own (like the thalamus's sorting of your
words) finds out which parts it wants: "bau dir einen körper mit torso, kopf, armen und beinen"
wants a torso, a head, two arms and two legs. On twelve goals in German and English it listed
the parts rightly for all nine that build something and nothing for the three that do not.
After every `build` the executive is told first how the model stands against that ("It has:
torso, head. Still missing: 2 × arm, 2 × leg"). When nothing is missing and the model is one
valid solid, the goal is done by reflex, with no model asked; if you asked for it, it is put on
your screen first and you are told, in the voice's language ("body ist fertig: alle Teile, die du
wolltest, sind da, und es ist ein einziger fester Körper. Ich habe es dir in FreeCAD geöffnet.").
The long work is held to the same list: once its body is whole the wandering mind is told so,
and not to rebuild it for its own sake.

A whole model does not get worse by a slip: a part of the body given again as a hole under its
own name is taken for what it was meant to be, a part (the model did that five times in three
trials), and a change after which a model that was one valid solid falls into pieces is not
kept: the model is put back as it stood, and the mind is told what the change did. In the live
trials asked for a body, it built all six parts as one solid, showed them and said so 81 s and
about four minutes after being asked.

## Foresight

Every tract also learns what the steps that set out along it came to: for the neurons a step
set out from, what that step went on to earn (`Net.foresee`). Before the executive's step goes
out to the motor cortex, its neurons are asked what it will come to; where they foresee it
failing (-0.25 or worse), the critic tells the executive so, with the step like it that went
badly and its lesson, and the executive decides once more. Replayed over this brain's history,
asked before each step: of the steps forewarned that way, 89 in 100 went on to be punished (174
of 1,044 steps), where of all steps 69 in 100 were. Right after a step that worked the neurons
are not asked: there they were right only 6 times in 10. It costs one more decision on about
one step in six. A brain whose tracts are new learns this from its past when it relives it.

Measured and left alone: acting by habit. The motor cortex could have made the routine calls
(read a file, run a script, list the files) with no model, from what its tract brings up. Under
the rules that would make it safe (never the same call twice, never right after a failure, the
file named in the step itself) the habit got the whole call right 72 times in 100, for 4 in 100
actions, and 37 in 100 in the last third of the brain's life; and those calls take the model a
median 3.7 s anyway. Not worth it, so the model still makes every call.

## It thinks on: an inner voice, wants of its own

The thinker thinks when it has something to decide; between, and beside it, there is now an
inner voice. Every few seconds a small model on the processor (the eyes model, `qwen3.5:2b`, set
as `models.inner`; 34 tokens a second here, 4 s a thought) says one short thing to itself in the
first person, goal or no goal: "That terminal window feels so empty, just a raw stream of data and
no human face staring back at me." It does not act. It is heard: the executive gets its last two
murmurs with every step, the wandering mind goes on from the last one when its thought is free,
and the stream shows it as `… inner voice`. Murmurs are not remembered one by one: one every ten
seconds would bury everything else.

Its attention moves on by itself: one murmur on the work in hand, the next on what it wants, the
next on what is around it (what its eyes last saw), the next on a question or interest of its
own. Left to go over the same ground, it talked itself into a spiral in its first trial: over a
file that was never a model, it told itself eight times in six minutes to wipe its workspace
and start over. With its attention turning, and the machine's facts about what it dwells on
beside it, it called that same file "just noise" instead.

What it wants is worked out from how things stand, with no model: the human's company (the
longer you are silent), something that works (the more of its last steps failed), something new
(the more its last thoughts went over the same ground), and rest. The strongest is told to the
inner voice and to the wandering mind. A thought that is free, one in four when it has a long
work, every one when it has none, is told it is its own to choose: what it wants, what it has
been getting better at (goals alike in their patterns, of which the later ended better than the
earlier: curiosity has been found to follow learning progress, not mastery and not failure), or
one of its open questions. A question it thought of and did not think through is kept, and
stays open until the regions have thought it through.

And it learns how well it knows itself. With every step the executive says how sure it is that
the step will work, 0 to 100; when the result comes, the two are put side by side, and once
there are ten it is told with every step: "when you said you were 70 or more sure, the step
worked 9 of 30 times". Whether its confidence comes to mean something is the one piece of
self-awareness here that can be measured. Press `m` to read all of it: what it wants, its inner
voice, its interests, its open questions, and how well it knows itself.

## Night school: the thinker itself learns

Everything above steers a model that never changes. Night school changes it. Every reply the
model gives a region is kept with what it was shown (`experience/`, a file a day), beside the
memory it led to. In sleep, once there are enough (30 good and 30 bad by default), each is
labelled by what that memory came to: good above +0.15, bad below -0.15. That value is the
critic's judgment of how the mind went at it and what was measured, half and half, with credit
flowing back from what it led to, so an honest "this cannot be done, because" can count as good
and a claim of success that the run did not bear out as bad. Being right is not the only thing
that pays.

Then the machine that launched the brain teaches (`school.py`, `train.py`): the model server lets
go of the graphics card, a training container learns a LoRA adapter from the day's replies with
KTO (which learns from answers marked good or bad, which is what the brain has), the adapter is
folded into a copy of the model and made as small as the model the server runs, and the server
is given it as a model of its own (`granite4.1-night-N`). Then the model as it was and the model
as it now is sit the same exam, 28 questions with known answers (sorting what a human says,
naming the parts a goal wants). What does worse by more than one answer is thrown away. What
passes becomes the model the brain thinks with; every night learns afresh on top of the model
it started with, so nights never pile on one another. The brain sleeps on until it is done.

Measured here: 120 lessons took 13 minutes on the RTX 5060 and 6.2 GB of its memory, folding
and shrinking 5 minutes more, the exam under a minute; a night of 400 takes about 50. The model
of that first night scored 26 of 28 on the exam, as the model before it did, and in a live trial
built the body it was asked for, showed it and said so 91 s after being asked: it thinks as well
as before. That night's lessons were taken from old history; whether nights of its own make it
better is still to be seen. What it needs on this machine: Docker with the graphics card, the trainer
image (`docker build -t cortex-trainer trainer`, about 23 GB), and the full model to learn on in
`~/.cortex/models/granite-4.1-8b` (17.6 GB, from Hugging Face). Without them a night says why
and the brain wakes as it was. Switch it off with `night_school: false` under `learning`.

The exam only shows that a night did not make it forget what it could. Whether a night made it
better shows in the days after, in what its steps earn. One night is little: KTO's loss stayed at
0.50 over the first 120 lessons, where it starts. It is meant to add up over many nights.

## What you raise comes first

The model has one slot, and calls wait in line for it in three ranks: your words (the thought in
the making is dropped and thought again), then what you raised (the regions' turns and the
conclusion on a matter about itself, and every step of work you asked for: next in line, nothing
dropped), then everything else. A region busy with work of its own is not waited for while
another can speak. A reply cut off at the token limit is said for what it is ("it ran out of room
after 2000 tokens while it was being written: write the file shorter, or in two files") and the
executive hears of it, where before it was a parser's error; on this brain that cost 17 of 1,638
actions.

What it did not speed up: asked to give itself a name, it answered 29 s after the words in both
trials since, as before. That time is four model calls of about 5 s each (three regions' turns
and the conclusion) and the executive finishing the step it was on.

The price: a matter for the mind itself takes 24 to 31 seconds to be answered, in two parts; a
heartbeat costs under a millisecond where it cost 40 microseconds; `synapses.npz` grew from 0.9
to 1.3 MB on this brain, and the one from before is kept beside it as
`synapses.npz.before-tracts` the first time a brain wakes with tracts.

## How it learns

Nothing is erased for being wrong. A reward does three things:

1. **It values the thought.** Rewarded memories surface more readily in recall, under "worked
   before"; punished ones surface too, under "went badly before", with the critic's lesson attached.
2. **It adjusts heuristics.** Each region carries short learned rules that are put into its prompt.
   The rules that were active when a thought was produced gain or lose weight by the *prediction
   error*: reward minus what that region usually earns. Average results change nothing; better than
   expected reinforces. Rules that keep losing are suppressed, not deleted. Which rules a region is
   handed is decided by their patterns: a rule about a theme the mind once dwelt on speaks up only
   when the mind is on that theme again, and of several wordings of one rule only the strongest
   is heard. On this brain 14 of 28 rules were about thread settings in pandas, a theme that 260
   of its 603 goals and thoughts share 60% of their pattern with; a general rule ("Run code
   before claiming it works") shares that much with none of them.
3. **It travels back.** The reward flows up the chain of parent thoughts with a discount, so the
   idea behind a good outcome gets credit, not only the last step.

The critic's opinion is not trusted alone. Half of an action's reward (configurable) comes from what
was *measured*: did the script run, did it crash, did the write change anything, was the action an
exact repeat. Two actions in a row that failed or repeated, or two runs in a row that failed
whatever was rewritten in between, call the regions together to think about why; if it goes on
failing after that, the goal ends. "Done" is a claim: when the last run failed or built nothing
valid, or code was written and never run, the critic says so and the executive decides once
more; and a model built because you asked is not done until `show` has put it on your screen.
Dopamine is the running
prediction error: when things go worse than expected, temperatures rise and the brain explores more.

The model is asked only where words are needed. A plan is not judged for how it reads, and
neither is a conclusion or a region's turn in a thinking-through: they earn what the action they
led to earns, through the reward that travels back. A repeat is punished by reflex.

## Models

Chosen for this machine (RTX 5060 8 GB, 32 GB RAM) by research and then measured with this code:

| role | model | measured |
|---|---|---|
| thinking and critic | `granite4.1:8b`, context 6144 | 71-73 tokens/s, 100% on GPU |
| embeddings | `qwen3-embedding:0.6b`, on CPU | about 20 ms per embedding |

- At context 8192 Granite spills 8% to the CPU and drops to 57 tokens/s; 6144 is the sweet spot.
  At 16144 a quarter of it is on the CPU and it writes 29 tokens/s: the same brain took 2.3
  actions a minute there and 4.8 at 6144. The stream says so in red when the model does not fit.
- `qwen3:8b` works too, at the same speed (72 tokens/s, context 8192). In five-minute trials both
  finished about two goals in three; Granite chose checkable code goals more consistently, Qwen3
  drifted toward journal-writing. A handful of runs: an indication, not a benchmark.
- One chat model serves every region. On 8 GB a second chat model would evict the first on each call.
- The server handles one request at a time by default. To let two regions think at once, set the
  user environment variables `OLLAMA_NUM_PARALLEL=2` and `OLLAMA_KV_CACHE_TYPE=q8_0`, restart
  Ollama, and set `parallel: 2` in `brain.yaml`. Unmeasured here; check with `cortex doctor`.
  `parallel` above what the server allows gains nothing: the calls queue up there instead, where
  your words can no longer go first.

The full research report is in `C:\Users\User\reports\Local LLMs for 8 GB agents.md`.

## Safety

- Files: every path is resolved and refused if it leaves `workspace/`.
- Code: scripts run in a separate Python process with a timeout and an output cap. Imports are
  checked against an allow-list (`safety.allowed_imports` in `brain.yaml`) before a script runs, and
  `open()` is confined to the workspace while it runs. No network, no subprocess, no `os`.
- Voice: speech runs outside the workspace, from fixed code. Its words reach that code as data,
  at most 600 characters at a time, and are never executed.
- Eyes: the titles of your open windows reach the brain, are put into what the model is shown and
  are kept in `memory.db` when one of them is remarked on. A title can hold the name of a
  document, a page or a person. With the amygdala off the brain's scripts can use the network
  (see *Full mode*), so switch `allow_screen` off if what is on your screen is nobody's business.
- Screen: `show` hands a file to your browser and nothing comes back from it. Only pages and
  pictures inside the workspace qualify; guarded, a web address is refused. A model is opened in
  FreeCAD by a fixed script that is given the file's path: a `.FCStd` is data, and nothing the
  brain wrote is run on this machine. Keep FreeCAD up to date all the same, as for any program
  that opens files from elsewhere.
- FreeCAD: scripts for `run_freecad` are checked like any other, with FreeCAD's own modules
  allowed, and run only inside the container.
- Its own source is readable to it, never writable. It improves through heuristics, not by
  rewriting itself.

This is a guard against a small model wandering off by accident. It is **not** a hard sandbox:
code written deliberately to escape could. The container is the wall behind it: of this machine's
files, only the brain's folder exists in there.

### Full mode

Switch the Amygdala off (`e`, or `enabled: false` under `regions.amygdala` in `brain.yaml`) and
the guards above are gone from its next action on: scripts import anything, start programs, use the
network and `pip install`; paths may lead anywhere. The state panel shows `mode full` while it lasts.

- It only takes effect in the container. With `--local` the guards stay on whatever the switch says.
- The brain is told it is free, but its drives still say "stay inside your workspace" and its
  memories still hold the lessons from being blocked. Edit the drives (`s`) if you want it to roam.
- What it can still reach: its own folder (so it can rewrite its `brain.yaml` and `memory.db`), the
  internet, and whatever listens on this machine's network, Ollama included. It can also open
  any web address in your browser with `show`.
- What it installs or changes outside its folder is gone when the container stops.
- The timeout, the output cap and the file size limit still apply; they are in settings.

## Develop

```powershell
uv run pytest      # runs the whole brain against a scripted model; no model server needed
```
