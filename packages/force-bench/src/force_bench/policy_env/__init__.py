"""ContactBench rows as policy-driven episodes (clockwork-5 fork): ``episode.run_row`` builds a row's own scene,
drives it frame by frame with a driver, and judges it on world state (``adapters``), with the row's upstream
verdict on the same rollout where that verdict does not read the expert's internal state.  See JUDGE.md."""
