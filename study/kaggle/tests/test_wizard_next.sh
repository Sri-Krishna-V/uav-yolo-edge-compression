#!/usr/bin/env bash
# Drive wizard.sh's next_flow through the broken-chain cases with Kaggle, git and prompts stubbed out.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" || exit 1
T=$(mktemp -d)
head -n -1 study/kaggle/wizard.sh > "$T/w.sh"   # everything but the final run_wizard call
DOWNLOADS_BEFORE=$(ls study/kaggle/.downloads 2>/dev/null || true)  # fetch_output downloads into the repo's own folder
pass=0; fail=0

scenario() {  # scenario NAME LAST_NOTEBOOK ENV_EXTRA : runs next_flow in a subshell, output in $T/NAME.log
  local name="$1" last="$2" extra="$3"
  printf "LAST_NOTEBOOK='%s'\n%s" "$last" "$extra" > "$T/env-$name"
  (
    set -euo pipefail
    # shellcheck disable=SC1091
    source "$T/w.sh"
    ENV_FILE="$T/env-$name"; WRITTEN_ENV=()
    pause() { :; }; must() { :; }; confirm() { return 1; }; open_url() { :; }; sync_repo() { return 0; }
    install_cli() { :; }; clipboard() { cat >/dev/null; }; _clear() { :; }; wrap_up() { :; }; git() { return 0; }
    kaggle_cli() {
      local slug="${3##*/}" v
      v="STATUS_${slug//-/_}"
      case "$1 $2" in
        "kernels status") printf '%s has status "%s"\n' "$3" "${!v:-KernelWorkerStatus.COMPLETE}" ;;
        # A version still running has no output yet. Like the real CLI (kaggle 2.2.4 kernels_output), every download
        # also writes the version's log as <slug>.log, whatever --file-pattern says.
        "kernels output")
          if [[ "$6" == --file-pattern ]]; then printf '%s %s\n' "$slug" "$7"; else printf '%s WHOLE-OUTPUT\n' "$slug"; fi >> "$T/calls"
          [[ "${!v:-}" == *RUNNING* ]] && return 0
          [[ -d "$T/out/$slug" ]] && cp -r "$T/out/$slug/." "$5/"
          printf 'log\n' > "$5/$slug.log"; return 0 ;;
      esac
    }
    for k in SID_A SID_B; do v=$(_existing "$k" || true); printf -v "$k" '%s' "$v"; done
    KAGGLE_TOKEN=x KAGGLE_WHO=vid KAGGLE_OWNER=teammate KAGGLE_MIRROR="$T/mirror" SESSIONS_STARTED=3
    NOTEBOOK_A=https://www.kaggle.com/code/vid/uav-vid-a NOTEBOOK_B=https://www.kaggle.com/code/vid/uav-vid-b A_READS_B=yes
    mkdir -p "$KAGGLE_MIRROR"
    LAST_NOTEBOOK="$last"
    next_flow
  ) > "$T/$name.log" 2>&1
  echo $? > "$T/$name.rc"
}

check() {  # check LABEL CONDITION...
  local label="$1"; shift
  if "$@"; then pass=$((pass + 1)); echo "PASS $label"; else fail=$((fail + 1)); echo "FAIL $label"; fi
}
envval() { grep -E "^$2=" "$T/env-$1" | tail -n1 | cut -d"'" -f2; }
cell() { grep '^TASK' study/kaggle/.cell-to-paste.txt; }

# S1: Vidhyarth's case. A's newest version saved no bring-home -> no dead end; rerun in A; no --input-from yet.
mkdir -p "$T/out"
scenario s1 A ""
check "S1 no bring-home: wizard goes on" grep -q "saved nothing" "$T/s1.log"
check "S1 next session in notebook A again" grep -q "Get notebook A ready" "$T/s1.log"
check "S1 input to check is uav-vid-b" grep -q "uav-vid-b must be listed" "$T/s1.log"
check "S1 TASK is the plain next task" [ "$(cell)" == 'TASK = "train T2 P4 --new T2 P4"' ]
check "S1 LAST_NOTEBOOK stays A" [ "$(envval s1 LAST_NOTEBOOK)" == "A" ]
hint_quotes_session() {  # the message the P/D/C hint tells the person to look for is one session.py prints
  local q
  q=$(grep -o '"[^"]*best\.pt[^"]*"' "$T/s1.log" | head -n1 | tr -d '"')
  [[ -n "$q" ]] && grep -qF "$q" study/kaggle/session.py
}
check "S1 the best.pt hint quotes what session.py prints" hint_quotes_session
check "S1 never downloads the whole output of a version without bring-home" bash -c "! grep -q WHOLE-OUTPUT '$T/calls'"

# S2: a normal session in B with a stamp -> SID_B saved, next in A, TASK carries --input-from.
mkdir -p "$T/out/uav-vid-b/bring-home"
printf 'T2 yolo26m-640-s0: PAUSED: resume it next session\n' > "$T/out/uav-vid-b/bring-home/SESSION_RESULT.txt"
printf '{"session_id": "20261022-010000-abcd", "result": "ok"}' > "$T/out/uav-vid-b/bring-home/session_stamp.json"
scenario s2 B ""
check "S2 SID_B remembered" [ "$(envval s2 SID_B)" == "20261022-010000-abcd" ]
check "S2 next in A" grep -q "Get notebook A ready" "$T/s2.log"
check "S2 TASK has --input-from" [ "$(cell)" == 'TASK = "train T2 P4 --new T2 P4 --input-from 20261022-010000-abcd"' ]
check "S2 LAST_NOTEBOOK becomes A" [ "$(envval s2 LAST_NOTEBOOK)" == "A" ]

# S3: a session is already running in the notebook the wizard would use -> wait, never start a second one.
scenario s3 A "" 2>/dev/null
STATUS_uav_vid_b=KernelWorkerStatus.RUNNING; export STATUS_uav_vid_b
scenario s3 A ""
unset STATUS_uav_vid_b
check "S3 running session detected" grep -q "already running in notebook" "$T/s3.log"
check "S3 exits cleanly" [ "$(cat "$T/s3.rc")" == "0" ]
check "S3 LAST_NOTEBOOK becomes B" [ "$(envval s3 LAST_NOTEBOOK)" == "B" ]
check "S3 no session started" bash -c "! grep -q 'Start the next session' '$T/s3.log'"

# S4: the session in A stopped in preflight (stamp 'stopped') -> rerun in A; its input B must be SID_B's version.
mkdir -p "$T/out/uav-vid-a/bring-home"
printf 'STOPPED before training: The input is not the session the wizard just brought home.\n' > "$T/out/uav-vid-a/bring-home/SESSION_RESULT.txt"
printf '{"session_id": "20261022-020000-ffff", "result": "stopped"}' > "$T/out/uav-vid-a/bring-home/session_stamp.json"
scenario s4 A "SID_B='20261022-010000-abcd'
"
check "S4 stopped -> rerun in A" grep -q "Get notebook A ready" "$T/s4.log"
check "S4 stopped stamp not remembered as SID_A" [ -z "$(envval s4 SID_A)" ]
check "S4 TASK checks B's version" [ "$(cell)" == 'TASK = "train T2 P4 --new T2 P4 --input-from 20261022-010000-abcd"' ]

# S5: a BLOCKED result that names wizard.sh repair -> the wizard points at the Input first, then repair.
rm -rf "$T/out/uav-vid-a"; mkdir -p "$T/out/uav-vid-a/bring-home"
printf "T2 yolo26m-640-s0: PAUSED: resume it next session\nP4 yolo26s-960-s0-prune30: BLOCKED: B2's best.pt is missing. ... bash study/kaggle/wizard.sh repair\n" > "$T/out/uav-vid-a/bring-home/SESSION_RESULT.txt"
printf '{"session_id": "20261022-030000-1111", "result": "ok"}' > "$T/out/uav-vid-a/bring-home/session_stamp.json"
scenario s5 A ""
check "S5 repair hint shown" grep -q "wizard.sh repair" "$T/s5.log"
check "S5 next in B" grep -q "Get notebook B ready" "$T/s5.log"

# S6: notebook B ran a later session the wizard never noted (LAST_NOTEBOOK still says A) -> B is the last session.
printf '{"session_id": "20261022-050000-2222", "result": "ok"}' > "$T/out/uav-vid-b/bring-home/session_stamp.json"
scenario s6 A ""
check "S6 unnoted later session in B detected" grep -q "Notebook B ran a session after notebook A" "$T/s6.log"
check "S6 brings B home" grep -q "Asking Kaggle about vid/uav-vid-b" "$T/s6.log"
check "S6 next in A, reading B's new version" [ "$(cell)" == 'TASK = "train T2 P4 --new T2 P4 --input-from 20261022-050000-2222"' ]

# S7: the person comes back too early: the last session (B) is still running, A holds the session before it.
# The wizard must say so and wait, not take A for the last session and start one in the running notebook.
STATUS_uav_vid_b=KernelWorkerStatus.RUNNING; export STATUS_uav_vid_b
scenario s7 B ""
unset STATUS_uav_vid_b
check "S7 says the session is still running" grep -q "still running" "$T/s7.log"
check "S7 exits cleanly" [ "$(cat "$T/s7.rc")" == "0" ]
check "S7 LAST_NOTEBOOK stays B" [ "$(envval s7 LAST_NOTEBOOK)" == "B" ]
check "S7 no unnoted-session warning" bash -c "! grep -q 'ran a session after' '$T/s7.log'"
check "S7 no session started" bash -c "! grep -q 'Start the next session' '$T/s7.log'"

echo; echo "$pass passed, $fail failed"; [[ $fail -eq 0 ]] || { echo "logs in $T"; for f in "$T"/s*.log; do echo "== $f"; tail -25 "$f"; done; }
rm -f study/kaggle/.cell-to-paste.txt
for d in study/kaggle/.downloads/*-uav-vid-[ab]; do  # the scenarios' downloads, never the person's own
  [[ -d "$d" ]] && ! grep -qxF "${d##*/}" <<< "$DOWNLOADS_BEFORE" && rm -rf "$d"
done
[[ $fail -eq 0 ]]
