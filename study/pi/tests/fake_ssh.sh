#!/usr/bin/env bash
# Stands in for ssh in the wizard's rehearsal: the "Pi" is the folder $FAKE_PI_HOME on this machine, and the remote
# command runs there in bash, with HOME pointing at it. -n reads no stdin, as ssh -n does; the other options are ignored.
no_stdin=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) no_stdin=1; shift ;;
    -o) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
shift   # user@host
cd "$FAKE_PI_HOME" || exit 255
if [[ -n "$no_stdin" ]]; then HOME="$FAKE_PI_HOME" exec bash -c "$*" < /dev/null; fi
HOME="$FAKE_PI_HOME" exec bash -c "$*"
