# Power and energy per inference on the Jetson Orin Nano 8 GB

Board input power (tegrastats VDD_IN, 100 ms samples), deployed runtime through launch.sh (jetson_clocks on),
10 reference frames, back-to-back predictions. Energy = mean VDD_IN x wall time per prediction. Cases: 4 pose goal,
6 image goal with a new goal every call, 6c image goal with an unchanged goal. Script: eval/jetson/power_measure.py.

**Pending: not run on the Jetson yet.** Run `./eval/jetson/run_power_sweep.sh` on the Jetson.
