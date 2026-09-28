# Fault-injection check of the ROS replay test suite (2026-09-28)

Faults injected one at a time into a COPY of the node (fault_injection/omnivla_nav_node_fault.py, OMNI_FAULT=...; the
deployed node deploy/ros2/omnivla_nav_node.py is unchanged, md5 5803f8f5...). Run by fault_injection/run_faults.sh on the
Jetson (no rover; localhost DDS, UDP sink). Suite = replay_test.py checks (+ verify_replay6.py for image goal).
Logs: results/fault_injection/ (round 2), results/faults_round1 on the Jetson (round 1).

## Round 1: the suite as it was (13 checks)
| Fault | Pose (mode 4) | Image goal (mode 6) |
|---|---|---|
| none (control) | 13/13 pass | 13/13 pass, chunks reproduced |
| 1 flipped steering sign (servo stage) | NOT CAUGHT (13/13 pass) | NOT CAUGHT (13/13 pass) |
| 2 stale goal | only turns_follow (statistical) | goal_cache_refresh + turns_follow |
| 3 chunks delayed 2 s | report CRASHED (TypeError) | inference_latency (by accident) + crash |
| 4 estop ignored | estop_neutral | estop_neutral |
| 5 wrong mode (4->8, 6->4) | chunks_match_validated | report CRASHED (KeyError); verifier crashed |

## Strengthened suite (17 checks + verifier)
New: servo_matches_command (servo pulses == ServoMap(cmd_vel) of the same tick), steer_follows_chunk (cmd_vel turn
direction == ChunkExecutor command from the latest chunk), goal_fresh (mode 4: <= 20% of chunks with a goal other than
the frame's), chunk_latency (frame->chunk p95 <= 1.5 s), image_goal_chunk_fields (mode 6), crash-proof report (an
exception is a FAIL), verifier counts missing fields / empty logs as mismatches.

## Round 2
| Fault | Pose (mode 4): failing checks | Image goal (mode 6): failing checks |
|---|---|---|
| none (control) | none (17/17) | none (17/17); verifier: all chunks bit-exact |
| 1 steer_flip | servo_matches_command (1660/1805 pairs differ) | servo_matches_command (1638/1804) |
| 1b cmd_flip (extra: controller output flipped) | steer_follows_chunk (0/903 agree) | not run |
| 2 stale_goal | goal_fresh (221/224), turns_follow_actual_path | goal_cache_refresh (1 goal change / 20 clips), turns_follow_actual_path |
| 3 delay2s | chunk_latency (p95 2.51 s), drives_when_healthy (0% RUN: chunks expire), resumes_after_stale, steer_follows_chunk | chunk_latency (p95 3.13 s), inference_latency, drives_when_healthy, resumes_after_stale, steer_follows_chunk |
| 4 estop_ignored | estop_neutral (97/97 non-neutral) | estop_neutral (95/95) |
| 5 wrong_mode | chunks_match_validated (max|d| 1.96, 0 exact) | image_goal_chunk_fields (225/225), goal_cache_refresh, steer_follows_chunk, turns_follow; verifier MISMATCH |

Margins in clean runs (real node, round 2): steer_follows_chunk 894/895 - 903/904 (pose), 732/732 - 799/799 (image);
servo_matches_command 0-1 mismatches of ~1800; goal_fresh 11-12/225; chunk_latency p95 0.51 s (pose) / 1.10-1.11 s (image).
Real-node replays with the strengthened suite: pose 17/17, image goal N=1 17/17, N=3 17/17; verifier bit-exact.

## Limits
- The verifier reproduces what the node computed (goal the node held), so by design it cannot catch a stale goal;
  goal_fresh / goal_cache_refresh do.
- Faults only tested one at a time; the ROS layer only (no rover, no real camera timing).
