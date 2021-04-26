#!/usr/bin/env bash

: '
I use this for parallel execution in separate windows, iTerm2 must be installed (Mac).
Call this file from terminal:
>> bash experiment_runner.sh
'

ssh -t luga 'bash /Net/Groups/BGI/people/bkraft/git/q10hybrid/experiments/run_exp_02.sh 0 --new_study'

osascript <<'END'
tell application "iTerm2"
    tell current session of current tab of current window
        write text "ssh -t luga 'bash /Net/Groups/BGI/people/bkraft/git/q10hybrid/experiments/run_exp_02.sh 0; bash -l'"
        split horizontally with default profile
        split vertically with default profile
    end tell
    tell second session of current tab of current window
        write text "ssh -t luga 'bash /Net/Groups/BGI/people/bkraft/git/q10hybrid/experiments/run_exp_02.sh 0; bash -l'"
    end tell
    tell third session of current tab of current window
        write text "ssh -t luga 'bash /Net/Groups/BGI/people/bkraft/git/q10hybrid/experiments/run_exp_02.sh 1; bash -l'"
        split vertically with default profile
    end tell
    tell fourth session of current tab of current window
        write text "ssh -t luga 'bash /Net/Groups/BGI/people/bkraft/git/q10hybrid/experiments/run_exp_02.sh 1; bash -l'"
    end tell
end tell
END
