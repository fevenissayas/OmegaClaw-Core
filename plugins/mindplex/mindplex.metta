!(import! &self (library Omega ./plugins/mindplex/mindplex.py))

; Configuration (contextWindowTokens, maxOutputToken, maxNewInputLoops,
; wakeupInterval, maxFeedback) is read by mindplex.start on the Python side.
(= (loadOmegaPlugin)
    (progn
        (py-call (mindplex.start))
        (add-skill
            recall
            "Bring task evidence that was left out of context back on the next turn, by the exact ids shown in TOOL_RESULT_OMITTED or SOURCE markers, comma separated."
            (ids))
        (log INFO "mindplex-plugin" "Mindplex context compiler is enabled")))

(= (recall $ids)
    (py-call (mindplex.recall $ids)))
