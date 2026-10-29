#!/usr/bin/env bash
# Rehearse jetson/wizard.sh's stages 1-6 with this laptop standing in for the Jetson (#39):
#
#   bash jetson/tests/rehearse.sh
#
# pi/tests/fake_ssh.sh runs every remote command in a temporary "board home" here. Shims on PATH stand in for what the
# laptop has not got: a canned GO probe and board state (python3), a complete install (the venv's python), no desktop
# (systemctl), and a gh that is never signed in, so nothing is posted to GitHub. The checkpoints are small fakes in a
# temporary folder, one of them as a best.pt.zip. Stages 1-6 run for real: the files, the copy over the fake ssh and
# its hash checks. Stage 7 is declined, because the laptop cannot build a TensorRT engine; the engines, the session
# and the copy home are the board's first real test. Then a second run must find everything already copied.
set -euo pipefail
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP=$(mktemp -d)
export FAKE_PI_HOME="$TMP/board-home" WIZ_SSH="$STUDY/pi/tests/fake_ssh.sh" WIZ_RESULTS="$TMP/home" ENV_FILE="$TMP/env" \
  WIZ_PROBE_OUT="$TMP/jetson_probe.txt"
PY="$STUDY/.venv/Scripts/python.exe"; [[ -x "$PY" ]] || PY="$STUDY/.venv/bin/python"
BIN="$TMP/bin"
mkdir -p "$FAKE_PI_HOME/uav-venv/bin" "$BIN"
echo "rehearsal in $TMP"

# The board's python3: a canned probe and board state; anything else is the study's venv
cat > "$TMP/probe.txt" <<'EOF'
=== Jetson probe (study/scripts/jetson_probe.py) ===
board         NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super (REHEARSAL)
memory        7620 MB total
L4T           # R36 (release), REVISION: 4.4
JetPack       6.2 (from L4T)
TensorRT      10.3.0.30

VERDICT       GO
run mode      sudo nvpmodel -m 2   (MAXN_SUPER; it may ask to reboot), then sudo jetson_clocks
EOF
cat > "$BIN/python3" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" == scripts/jetson_probe.py ]]; then cat "$TMP/probe.txt"; exit 0; fi
if [[ "\${1:-}" == - ]]; then
  src=\$(cat)
  if [[ "\$src" == *board_state* ]]; then printf 'mode=2\nmode_name=MAXN_SUPER\nhigh=2\nhigh_name=MAXN_SUPER\nclocks=True\n'; exit 0; fi
  exec "$PY" "\$@" <<< "\$src"
fi
exec "$PY" "\$@"
EOF
# The venv's python: install.sh --check sees every pin it wants (read from the copied install.sh)
cat > "$FAKE_PI_HOME/uav-venv/bin/python" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" == - ]]; then
  src=\$(cat)
  if [[ "\$src" == *'for name in ("torch"'* ]]; then
    bash -c 'eval "\$(grep -E "^(NUMPY|ORT|WANT)=" "\$HOME/uav/study/jetson/install.sh")"; echo "\$WANT"'
    exit 0
  fi
  exec "$PY" "\$@" <<< "\$src"
fi
exec "$PY" "\$@"
EOF
printf '#!/usr/bin/env bash\ncase "$*" in "is-active display-manager") echo inactive; exit 3 ;; get-default) echo multi-user.target ;; esac\n' > "$BIN/systemctl"
printf '#!/usr/bin/env bash\nexit 1\n' > "$BIN/gh"
chmod +x "$BIN"/* "$FAKE_PI_HOME/uav-venv/bin/python"
export PATH="$BIN:$PATH"

# Fake checkpoints: three as best.pt, YOLO26s at 960 as a best.pt.zip in torch's zip layout
CK="$TMP/ckpt"
for r in yolo11s-640-s0 yolo26s-640-s0 yolo11s-960-s0 yolo26s-960-s0; do mkdir -p "$CK/$r"; done
for r in yolo11s-640-s0 yolo26s-640-s0 yolo11s-960-s0; do printf 'fake checkpoint %s\n' "$r" > "$CK/$r/best.pt"; done
"$PY" -c 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1], "w"); z.writestr("best/data.pkl", b"fake"); z.close()' \
  "$(cygpath -m "$CK/yolo26s-960-s0/best.pt.zip" 2>/dev/null || echo "$CK/yolo26s-960-s0/best.pt.zip")"

# Enter (banner); the checkpoint folder; y (copy the .zip); y (ready); the host and the user; y (not an Orin Nano:
# go on); Enter through stages 2-5; y (copy); Enter; n (start the builds)
printf '\n%s\ny\ny\nfakejet\ndev\ny\n\n\n\n\ny\n\nn\n' "$CK" | timeout 900 bash "$STUDY/jetson/wizard.sh" | tee "$TMP/wizard.out" || true

B="$FAKE_PI_HOME/uav/study"
fail=0
expect() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
expect "the .zip checkpoint was copied to best.pt" "[[ -f '$CK/yolo26s-960-s0/best.pt' ]]"
expect "the scripts and the plan reached the board" "[[ -f '$B/scripts/studysession.py' && -f '$B/plans/jetson.json' && -f '$B/jetson/export.sh' && -f '$B/cfg/invariants.yaml' ]]"
expect "the probe was saved on the laptop" "grep -q 'VERDICT       GO' '$WIZ_PROBE_OUT'"
expect "the images reached the board" "[[ \$(ls '$B/datasets/VisDrone/images/test' | wc -l) -eq 1610 && \$(ls '$B/datasets/VisDrone/labels/val' | wc -l) -eq 548 ]]"
expect "the drift reference reached the board" "[[ -f '$B/exports/yolo11s-640-s0/640/onnxruntime-fp32/manifest.json' ]]"
expect "the 4 checkpoints reached the board" "[[ \$(ls '$FAKE_PI_HOME'/uav-ckpt/*/best.pt | wc -l) -eq 4 ]]"
expect "the wizard reached stage 7 and stopped there" "grep -q 'No engine is built yet: 0 of 8' '$TMP/wizard.out'"
expect "nothing was offered for GitHub (gh is not signed in here)" "! grep -q 'Post the probe' '$TMP/wizard.out'"

# A second run finds everything copied. Its answers: no .zip to copy now, and nothing to confirm in stage 6
printf '\n%s\ny\nfakejet\ndev\ny\n\n\n\n\n\nn\n' "$CK" | timeout 600 bash "$STUDY/jetson/wizard.sh" > "$TMP/wizard2.out" 2>&1 || true
expect "a second run copies nothing again" "grep -q 'already there: datasets/VisDrone/images/test' '$TMP/wizard2.out' && ! grep -q 'copying ' '$TMP/wizard2.out'"
expect "a second run finds the checkpoints there" "[[ \$(grep -c 'already there: ~/uav-ckpt/' '$TMP/wizard2.out') -eq 4 ]]"
expect "a second run reaches stage 7 again" "grep -q 'No engine is built yet: 0 of 8' '$TMP/wizard2.out'"
(( fail == 0 )) || { echo "rehearsal FAILED: see $TMP"; exit 1; }
rm -rf "$TMP"   # it holds a copy of the test-dev images
echo "rehearsal passed"
