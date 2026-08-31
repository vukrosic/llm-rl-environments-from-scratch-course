# Source ledger

Checked 2026-08-31. Primary sources are preferred. The two paper figures in `assets/` are included for criticism, explanation, and research teaching; attribution remains visible on-slide.

## S01 — Apodex 1.1 paper

- Apodex Team, “Apodex 1.1: Training Deep Research Agents via Self-Evolving Environments,” arXiv:2608.23283 (2026): https://arxiv.org/html/2608.23283
- Used for the task formalism, environment families, PIVOT-RL description, agent-team architecture, and reported benchmark results.
- Boundary: the deck labels author-reported results and does not imply independent reproduction.

## S02 — FrontierAgent runtime

- Official repository: https://github.com/ApodexAI/FrontierAgent
- Used for the publicly released agent runtime, ReAct/Agent Team modes, evaluation workflow, and trajectory support.

## S03 — Released model

- Official model card, `apodex/Apodex-1.1-mini`: https://huggingface.co/apodex/Apodex-1.1-mini
- Used for the public 36B Mini checkpoint and Apache-2.0 release boundary.

## S04 — FrontierChallenge tasks

- Official dataset card: https://huggingface.co/datasets/apodex/FrontierChallenge
- Used for the 97-task count, 81 open-image tasks, 16 ORCA definitions, encrypted reference/grader split, and image availability caveat.

## S05 — Small executable examples

- Official examples: https://github.com/ApodexAI/executable-world-examples
- Used for lightweight inspection of task workspaces, actions, outputs, and trajectories without downloading large model weights.

## S06 — Harbor environment framework

- Official repository and documentation: https://github.com/harbor-framework/harbor
- Used for the portable task-package structure, Docker execution, agent trajectories, verifier output, and numeric reward convention.

## S07 — Tool use

- Schick et al., “Toolformer,” arXiv:2302.04761: https://arxiv.org/abs/2302.04761
- Used to motivate research on when a language model should call tools.

## S08 — Multi-turn RL environments

- Zhou et al., “RAGEN,” arXiv:2504.20073: https://arxiv.org/abs/2504.20073
- Official code: https://github.com/mll-lab-nu/RAGEN
- Used for initial-state diversity, interaction granularity, rollout-sampling hypotheses, the Sokoban environment visual in Figure 9, and the reward specification in Appendix C.1.

## S09 — Search RL

- Jin et al., “Search-R1,” arXiv:2503.09516: https://arxiv.org/abs/2503.09516
- Official code: https://github.com/PeterGriffinJin/Search-R1
- Used to motivate controlled research on search actions and retrieved evidence.

## S10 — Executable computer and coding benchmarks

- Xie et al., “OSWorld,” arXiv:2404.07972: https://arxiv.org/abs/2404.07972
- Jimenez et al., “SWE-bench,” arXiv:2310.06770: https://arxiv.org/abs/2310.06770
- Used to motivate stateful computer tasks and externally verified software tasks.

## S11 — Reproducible deep RL

- Henderson et al., “Deep Reinforcement Learning That Matters,” arXiv:1709.06560: https://arxiv.org/abs/1709.06560
- Used for seed, implementation, environment, and hyperparameter controls.

## S12 — OpenAI role specification

- Official live posting, “Research Engineer, Frontier Evals & Environments”: https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/
- Captured 2026-08-31 in `assets/openai-frontier-evals-role-full.png`; slide 2 uses the responsibility excerpt `assets/openai-role-core.png`.
- Used to map the course to RL environments, graders, measurement reliability, continuous evaluation, and concrete experiments.

## S13 — Anthropic role specifications

- Official live posting, “Research Engineer, Universes”: https://job-boards.greenhouse.io/anthropic/jobs/5061517008
- Official live posting, “Staff Software Engineer, Environments Infrastructure”: https://job-boards.greenhouse.io/anthropic/jobs/5367436008
- Captured 2026-08-31 in `assets/anthropic-universes-role-full.png`; slide 2 uses the role and responsibility excerpt `assets/anthropic-role-core.png`.
- Used to map the course to agentic training environments, rigorous evaluations, sandboxing, robust infrastructure, replay, and verification.

## S14 — ToolSandbox

- Lu et al., “ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities,” arXiv:2408.04682: https://arxiv.org/abs/2408.04682
- Official code: https://github.com/apple/ToolSandbox
- Used for the phone-state environment example: settings, contacts, messages, reminders, state dependencies, and milestone evaluation. Slide 5 reconstructs the cellular-off messaging example documented in Figure 1 and §2.3; it is not presented as a live model trajectory. Slide 6 charts the State Dependency values from Table 5 and visualizes the authors’ §4 explanation about erroneous parallel tool calls; values are author-reported 2024 results, not current rankings or our reproduction.

## S15 — tau-bench

- Yao et al., “tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains,” arXiv:2406.12045: https://arxiv.org/abs/2406.12045
- Current official implementation: https://github.com/sierra-research/tau2-bench
- Used for the airline customer-service example with domain policy, tools, tasks, and database-state evaluation.

## S16 — SWE-Gym

- Pan et al., “SWE-Gym: Training Software Engineering Agents and Verifiers,” ICML 2025: https://proceedings.mlr.press/v267/pan25g.html
- Official code: https://github.com/SWE-Gym/SWE-Gym
- Used for the repository-editing environment example with natural-language issues and executable tests.

## S17 — D3-Gym

- Moussa et al., “D3-Gym: Constructing Real-World Verifiable Environments for Data-Driven Discovery,” arXiv:2604.27977: https://arxiv.org/abs/2604.27977
- Official code and task environments: https://github.com/OSU-NLP-Group/D3-Gym
- Used for the scientific-data environment example: task instructions, datasets, executable dependencies, reference outputs, and evaluation scripts.

## S18 — CUA-Gym

- Wang et al., “CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents,” arXiv:2605.25624: https://arxiv.org/abs/2605.25624
- Used for the mock-browser environment example with co-generated task instructions, initial and golden application states, and executable reward functions.

## S19 — Agent Lightning v1.0

- He et al., “Agent Lightning v1.0: Towards Harnessed Agentic RL,” arXiv:2608.17528, August 2026: https://arxiv.org/abs/2608.17528
- Microsoft Research publication page: https://www.microsoft.com/en-us/research/publication/agent-lightning-v1-0-towards-harnessed-agentic-rl/
- Official open-source implementation: https://github.com/microsoft/agent-lightning
- Used on slide 6 to explain harnessed agentic RL: the deploy-time harness owns context construction, control flow, tool execution, and environment interaction, while the trainer observes the resulting LLM request-response pairs. The slide reconstructs the mechanism from Figures 1–2 and the Introduction; it does not reproduce the paper’s training result.
- Used on slide 7 for the paper’s capability-boundary curriculum in §5.1: run four pilot rollouts per coding task, remove tasks solved in all four attempts, keep mixed-success tasks, and add a smaller sample of all-failure tasks. The slide’s three task rows are explanatory examples, not paper data.

## S20 — EvoEnv

- Shi et al., “Learning to Build the Environment: Self-Evolving Reasoning RL via Verifiable Environment Synthesis,” arXiv:2605.14392, May 2026: https://arxiv.org/abs/2605.14392
- Used on slide 8 for the four-routine environment interface—sampler, oracle, renderer, and scorer—and the staged admission process covering execution, semantic review, solver-relative difficulty, and novelty. The graph reconstructs Figure 2, Appendix B, and Sections 3.1–3.4.
- Used on slide 9 for a simplified planted subset-sum environment based on Appendix D. The displayed numbers are ours; the solve–verify asymmetry and hidden-reference data flow come from the paper.

## S21 — Local tiny-Sokoban experiment

- Frozen protocol: `EXPERIMENT_PROTOCOL.md`
- Aggregate and per-seed results: `results/final/summary.json`
- Paired analysis and receipt audit: `results/final/analysis.json`
- Complete title replay: `artifacts/title_board_replay.json`
- Used for the trained replay on slides 1, 4, and 10; the frozen setup/result
  on slide 11; and the model/token explanation on slide 12.
- Boundary: the title state is excluded from aggregate metrics because it was
  inspected during pilots; the five-seed confirmation uses a separately frozen
  test split.

## S22 — EnvHarness

- Huang et al., “EnvHarness: Awakening Static Worlds for Agent Learning,”
  arXiv:2608.19880, August 2026: https://arxiv.org/abs/2608.19880
- Official project page: https://envharness.com/
- Official open-source implementation: https://github.com/google-research/envharness
- Used on slide 13 for a conceptual reconstruction of the project page’s
  designer-loop example: repeated untested submissions are diagnosed, a Rule
  component wraps `step()` to require testing, and the component is retained
  only after fresh rollouts change the behavior. The underlying task and
  verifier stay fixed.
- The ALFWorld comparison shown on the slide—85.4% for RL on the original
  environments and 88.3% for RL on EnvHarness environments—is reported by the
  authors; it is not a local reproduction.

## Claims we do not make

- We do not claim an exact Apodex training reproduction.
- We do not infer that Agent Team gains are caused only by architecture; model, prompt, budget, and harness details can confound comparisons.
- We do not equate verifier pass rate with scientific truth.
- Proposed follow-up experiments are hypotheses, not paper findings.
- Role alignment is a curriculum claim, not a guarantee of employment or an endorsement by OpenAI or Anthropic.
- We do not call the local experiment a numerical replication of RAGEN or
  general Sokoban competence.
