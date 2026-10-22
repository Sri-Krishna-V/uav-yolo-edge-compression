#!/usr/bin/env bash
# Rehearse pi/wizard.sh end to end with this laptop standing in for the Pi (#25):
#
#   bash pi/tests/rehearse.sh
#
# A fake ssh (fake_ssh.sh) runs every remote command in a temporary "Pi home" here, whose ~/sb/bin/python is the
# study's venv. The plan is plans/laptop-dry-run.json, with the runner's --dry-run: two 10-image cells and the two
# 200-image reference runs (about 2 minutes, 0.85 GB at peak on 2026-10-22). The results come home to a temporary
# folder, not to study/bench, and are scored there. Then a second run must find everything done. Answers are piped
# in, so the wizard never waits on a person.
set -euo pipefail
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP=$(mktemp -d)
export FAKE_PI_HOME="$TMP/pi-home" WIZ_SSH="$STUDY/pi/tests/fake_ssh.sh" WIZ_DRY_RUN=1 WIZ_RESULTS="$TMP/home" ENV_FILE="$TMP/env"
mkdir -p "$FAKE_PI_HOME/sb/bin"
PY="$STUDY/.venv/Scripts/python.exe"; [[ -x "$PY" ]] || PY="$STUDY/.venv/bin/python"
printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$PY" > "$FAKE_PI_HOME/sb/bin/python"
chmod +x "$FAKE_PI_HOME/sb/bin/python"
echo "rehearsal in $TMP"

# Enter (banner), y (ready), the host, the user, then y to everything: a y also passes a "Press Enter", and the
# progress loop takes one as a refresh. Six at once reach the start of the session; then one every 20 s, while the
# loop waits on the session, and for stage 7.
answers() { printf '\ny\nfakepi\ndev\n'; yes y | head -n 6; while :; do sleep 20; printf 'y\n'; done; }
answers | timeout 900 bash "$STUDY/pi/wizard.sh" plans/laptop-dry-run.json | tee "$TMP/wizard.out" || true

OUT="$WIZ_RESULTS/bench/laptop/laptop-dry-run"
fail=0
expect() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
expect "the files reached the fake Pi" "[[ -f '$FAKE_PI_HOME/uav/study/scripts/studysession.py' && -d '$FAKE_PI_HOME/uav/study/datasets/VisDrone/images/test' ]]"
expect "the session completed on the fake Pi" "grep -q '\"status\": \"complete\"' '$FAKE_PI_HOME/uav/study/bench/laptop/laptop-dry-run/session.json'"
expect "the results came home" "[[ -f '$OUT/summary.csv' ]]"
expect "both cells were scored at home" "[[ \$(find '$OUT/runs' -name score.json | wc -l) -eq 2 ]]"
expect "the wizard finished" "grep -q 'Setup complete' '$TMP/wizard.out'"

# A second run finds everything done: nothing is copied again, and the session is not run again.
{ printf '\ny\n\n\n'; yes y | head -n 12; } | timeout 300 bash "$STUDY/pi/wizard.sh" plans/laptop-dry-run.json > "$TMP/wizard2.out" 2>&1 || true
expect "a second run copies nothing again" "grep -q 'already there: datasets/VisDrone/images/test' '$TMP/wizard2.out' && ! grep -q 'copying ' '$TMP/wizard2.out'"
expect "a second run finds the session complete" "grep -q 'already complete on the Pi' '$TMP/wizard2.out' && grep -q 'Setup complete' '$TMP/wizard2.out'"
(( fail == 0 )) || { echo "rehearsal FAILED: see $TMP"; exit 1; }
rm -rf "$TMP"   # it holds a copy of the test-dev images
echo "rehearsal passed"
