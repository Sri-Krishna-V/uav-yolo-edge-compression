# Findings: the Galaxy M34 under Termux. What can block the session, and what quietly distorts its latency

Ticket: #27 (Pi 4B and phone sessions); the protocol is `study/PLATFORMS.md` § "Galaxy M34 (Termux, Python 3.14)" and
§ "Running a session". Plans: `study/plans/m34-speedtest.json`, `study/plans/m34-fullcell.json`.
Researched: 2026-10-25. Package indexes fetched 2026-10-25 (`termux-main` Release dated 2026-10-25 07:48 UTC, `termux-x11`
dated 2026-10-24 19:38 UTC).

**Question.** We run ONNX Runtime CPU sessions on a Galaxy M34 5G (Exynos 1280) under Termux, driven from the Windows
laptop over SSH with the Pi wizard. What can block the run, and what can distort its latency numbers without anyone
noticing? What do we set before the run, and what do we record?

**Verdict in one line:** install Termux and Termux:API from F-Droid (both from there), pin the default mirror, and
install the x11 opencv chain (about 190 MB download in all). Turn on "Disable child process restrictions". **Keep
Termux the visible, unlocked, screen-on foreground app for the whole session, even though you drive it over SSH.** On
One UI 8 a backgrounded Termux uid is moved within about a second to a little-core cpuset, and neither the wake lock
nor `taskset` stops that. Expect roughly 0.6–1.0 s per FP32 image and 20–45 min per plan. No Exynos 1280 YOLO number
is published; these are estimates.

---

## 1. Which Termux build

**The Play Store build is a separate fork with its own package repos.**
- It is Fredrik Fornwall's (@fornwall) fork in the `termux-play-store` GitHub org: *"Source for the current Termux
  build on Play Store"* ([org](https://github.com/termux-play-store)). Its packages come from a fork of termux-packages,
  *"Packages for Termux on Google Play"*, which *"regularly merges in changes"* from upstream
  ([termux-play-store/termux-packages](https://github.com/termux-play-store/termux-packages)). The README does not say
  where they are published. The user's install (`TERMUX_VERSION=googleplay.2026.10.09`) shows `https://termux.net
  stable/main` and `https://x11-packages.termux.net x11`, so in practice the Play build uses its own index.
- The Termux core team's announcement (agnostic-apollo, 8 Jun 2024): the Play build was *"not done by the current
  Termux core team"*; F-Droid and GitHub are *"the only official sources"*; Play releases stay deprecated and may be
  taken down; install from Play only *"at your own risk"*
  ([termux-app discussion #4000](https://github.com/termux/termux-app/discussions/4000)).
- It targets the latest Android SDK (bumped 2024-06-07), needs Android 11+, and removed `sharedUserId`. Many
  Termux:API commands have been built in since 2026-02-11, and *"The Termux:API app is not yet available"*
  ([org README](https://github.com/termux-play-store)).
- **Why our install failed (inferred):** its x11 index lagged. It offered `qt6-qt5compat` needing `qt6-shadertools
  (>= 6.11.1)` but not that package. The official x11 index of 2026-10-24 has the whole chain at 6.12.0 (§2). The
  `NO_PUBKEY 5A897D96E57CF20C` error for `packages.termux.dev` is expected: the Play build ships its own keyring, not the
  official one (**inferred**; I did not find this key documented).

**F-Droid or GitHub?** Both are official ([termux-app README](https://github.com/termux/termux-app)).
- F-Droid lists **0.118.3** (version code 1002) as "suggested", plus betas 0.119.0-beta.2 and beta.3. All need Android
  7.0+ and are signed by F-Droid ([f-droid.org/packages/com.termux](https://f-droid.org/en/packages/com.termux/)).
- GitHub's latest stable is also **v0.118.3 (22 May 2025)**, with v0.119.0-beta.3 the same day. Its APKs are signed with
  a **public test key** (*"This IS NOT an official developer key"*), so get them only from github.com/termux
  ([releases](https://github.com/termux/termux-app/releases), [README](https://github.com/termux/termux-app)).
- 0.118.3 bumps hiddenapibypass to fix a crash on *"Android 16 QPR1"* ([releases](https://github.com/termux/termux-app/releases)).
  The M34 is on Android 16 (§4.0), so use 0.118.3 or later, not an older APK.
- **Choose F-Droid** for the user's case. It is the source the README names first, it is the official build the
  maintainers recommend, and the user has already started there. GitHub would be equally valid; neither has a
  functional edge for us. The app has not been released since May 2025, so "stale" is not a concern of choice here.

**Termux:API must come from the same source.**
- Termux and its plugins share `sharedUserId` `com.termux` and *"must all be signed with the same key"*. Mixing sources
  fails with `INSTALL_FAILED_SHARED_USER_INCOMPATIBLE` or *"signatures do not match"*. To switch, uninstall **all**
  Termux apps and plugins first ([README](https://github.com/termux/termux-app)).
- So: Termux **and** Termux:API from F-Droid; then `pkg install termux-api` (the command-line half, 0.60.0 in today's
  index). `termux-battery-status` needs both.

**Known problems on Android 12+ / One UI 7–8.**
- README: *"Termux may be unstable on Android 12+"*: the phantom process killer (§3.1) can end processes with
  `[Process completed (signal 9) - press Enter]` ([README](https://github.com/termux/termux-app)).
- **One UI 8.x moves a backgrounded Termux to little cores** (§4.3) ([#5035](https://github.com/termux/termux-app/issues/5035),
  [#5086](https://github.com/termux/termux-app/issues/5086); both open, Mar–Apr 2026, no maintainer reply). This is the
  single biggest risk to the numbers.
- I found no install-time failure of the F-Droid build on Android 14–16 or One UI 6–8 beyond these.

---

## 2. Packages

**Versions in today's official indexes (aarch64):** python 3.14.8, python-numpy 2.5.4, onnxruntime / python-onnxruntime
1.31.0-3, opencv / opencv-python 4.14.0-4 (x11), qt6-qtbase, qt6-qt5compat, qt6-qtshadertools and qt6-qtdeclarative
6.12.0(-1), openssh 10.6p1-2, termux-api 0.60.0, python-pip 26.2.2-2. These match PLATFORMS.md's 2026-10-21 list.

**How they are built** (termux-packages `master`):
- `onnxruntime`: built from source with `-Donnxruntime_ENABLE_PYTHON=ON -Donnxruntime_USE_XNNPACK=ON
  -Donnxruntime_USE_NNAPI_BUILTIN=ON`, cross-compiled, the wheel `pip install`ed
  ([build.sh](https://github.com/termux/termux-packages/blob/master/packages/onnxruntime/build.sh)). `python-onnxruntime`
  depends on python, python-numpy and python-pip
  ([subpackage](https://github.com/termux/termux-packages/blob/master/packages/onnxruntime/python-onnxruntime.subpackage.sh)).
  Our bench asks for `CPUExecutionProvider` only, so XNNPACK and NNAPI are not used.
- `opencv`: built in x11-packages `-DWITH_QT=6`, depending on `qt6-qtbase, qt6-qt5compat, ffmpeg, libopenblas,
  ...` ([build.sh](https://github.com/termux/termux-packages/blob/master/x11-packages/opencv/build.sh)). `opencv-python` is
  its subpackage (`lib/python*`), so it pulls the whole Qt chain.
  `python-opencv-python` (x11, recommended by `opencv-python`) adds the pip `.dist-info` metadata only.
- `opencv-python`'s postinst warns if the installed numpy differs from the one it was built with
  ([subpackage](https://github.com/termux/termux-packages/blob/master/x11-packages/opencv/opencv-python.subpackage.sh)).

**Known problems:**
- `import onnxruntime` prints `UserWarning: Unsupported platform (android)`. It is harmless: upstream's way of saying
  "do not report Termux issues to us" ([termux-packages #31634](https://github.com/termux/termux-packages/issues/31634),
  closed 2026-09-16). It goes to stderr and so into `bench.log`. Do not treat it as a failure.
- After the Python 3.13 update, `import onnxruntime` failed with *No module named
  'onnxruntime.capi.onnxruntime_pybind11_state'*; fixed in March 2026
  ([#28769](https://github.com/termux/termux-packages/issues/28769)). The build.sh now installs that `.so` by hand. If it
  comes back, `pkg upgrade` first.
- **Partial upgrades break things.** `pkg install X` on an outdated install mixes old and new packages (numpy mismatch,
  `py3compile` crash: [#29145](https://github.com/termux/termux-packages/issues/29145), open). A Python minor bump
  (3.13 → 3.14) in a bulk upgrade broke a whole environment ([#30428](https://github.com/termux/termux-packages/issues/30428),
  open). So: `pkg upgrade` once, on the fresh install, before anything else, and **no `pkg upgrade` between the
  sessions of the study**.
- ORT's native on-device builds misdetect ARM64 and skip optimised MLAS kernels, per open
  [onnxruntime PR #29920](https://github.com/microsoft/onnxruntime/pull/29920). The Termux package is cross-compiled,
  so this should not apply (**unconfirmed**). A tell: an INT8 cell no faster than FP32 (§6).

**Download size of the opencv chain** (my computation from the official `Packages` indexes of 2026-10-24/25, dependency
closure of `opencv-python` minus the bootstrap and minus python/numpy/onnxruntime):
- **122 packages, 158 MB download, 835 MiB installed.** The largest: libllvm 30.7 MB (166 MiB installed, via mesa),
  qt6-qtdeclarative 16.9 MB, perl 15.4 MB, opencv 14.1 MB, qt6-qtbase 13.0 MB, ffmpeg 11.6 MB, libicu 10.2 MB.
- python + numpy + onnxruntime + openssh on top of the bootstrap: **23 packages, 31 MB download, 163 MiB installed.**
- **Total: about 190 MB download and about 1 GiB of storage**, plus the study files (exports and 301 MB of test-dev
  images) and about 1 GB for results.

**A lighter cv2?** None in Termux.
- No headless package exists in main, x11 or TUR (I listed the trees of `termux-user-repository/tur`: no opencv
  package). The request for `opencv-python-headless` ([#22822](https://github.com/termux/termux-packages/issues/22822),
  open since Jan 2025) got the answer that a headless variant *"would still be in the x11 repo since its parent package
  is"*. [#31513](https://github.com/termux/termux-packages/issues/31513) only adds headless pip metadata.
- PyPI has no Android wheels for opencv-python(-headless) 5.0.0.93, onnxruntime 1.31.0 or numpy 2.5.4 (checked the
  PyPI JSON on 2026-10-25), so `pip install opencv-python-headless` compiles OpenCV from source on the phone. A Jan 2025
  attempt failed ([#22822](https://github.com/termux/termux-packages/issues/22822)). Do not try it on session day.
- So the x11 chain is the way. It only costs download and storage: nothing in it runs during the bench.

**Mirrors: rotation and pinning** (termux-tools v1.45.0, which today's index ships as `termux-tools
1.46.0+really1.45.0`; [`pkg.in`](https://github.com/termux/termux-tools/blob/v1.45.0/scripts/pkg.in)):
- With no mirror chosen, `pkg` prints *"No mirror or mirror group selected. You might want to select one by running
  'termux-change-repo'"* and uses the default plus every regional mirror. It picks one at random, weighted, and
  re-picks when the cached index is more than **6 hours** old or the current mirror fails its check (L179–L333).
- The check only fetches `dists/stable/Release` and accepts any reply, so a stale mirror passes (**inferred** from the
  code). A rotation can then hand `apt` an older index than the last one. That is how you get "depends on X, but X is
  not going to be installed".
- **To pin the default mirror:** run `termux-change-repo` → "Single mirror" → the default (`packages-cf.termux.dev`,
  Cloudflare in front of `packages.termux.dev`). That writes `$PREFIX/etc/termux/chosen_mirrors` as a single file or
  symlink, and `pkg` then uses only it (L190–L196). Setting `TERMUX_PKG_NO_MIRROR_SELECT=1` additionally stops `pkg`
  from re-selecting while a mirror is set (L183–L186).

---

## 3. Keeping a long job alive

### 3.1 The phantom process killer (Android 12+)

- Android 12+ kills an app's child ("phantom") processes beyond **32 across all apps**, and kills a phantom process that
  uses *"too much CPU while its parent app process are also in the background"*, checked every **5 minutes**
  ([agnostic-apollo, Android-Docs](https://github.com/agnostic-apollo/Android-Docs/blob/master/en/docs/apps/processes/phantom-cached-and-empty-processes.md);
  no numeric CPU threshold is given). Termux's README warns of the resulting `[Process completed (signal 9)]`
  ([README](https://github.com/termux/termux-app)).
- **A daemonised `sshd` and everything it starts are phantom processes of Termux** (the doc's own example lists `sshd`
  in `dumpsys activity processes`). So is our `nohup python ... studysession.py`.
- **How to turn it off:**
  - **Android 14+ (the M34 on One UI 8):** Settings → About phone → Software information → tap *Build number* 7 times;
    then Settings → Developer options → **Disable child process restrictions** → on. Turning Developer options off again
    turns this off too ([Android-Docs](https://github.com/agnostic-apollo/Android-Docs/blob/master/en/docs/apps/processes/phantom-cached-and-empty-processes.md)).
    The toggle exists on One UI 8.5 (toggled in [#5035](https://github.com/termux/termux-app/issues/5035)). One user on
    One UI 6.1 reported it missing ([GrapheneOS forum thread](https://discuss.grapheneos.org/d/36033-should-i-disable-child-process-restrictions-in-developer-options),
    anecdotal).
  - **Fallback over adb (Android 12L and later):** `adb shell settings put global settings_enable_monitor_phantom_procs false`.
    Check with `adb shell settings get global settings_enable_monitor_phantom_procs` (expect `false`). Android 12.0 only
    had `device_config put activity_manager max_phantom_processes 2147483647` (+ `device_config set_sync_disabled_for_tests
    persistent`) ([Android-Docs](https://github.com/agnostic-apollo/Android-Docs/blob/master/en/docs/apps/processes/phantom-cached-and-empty-processes.md)).
    That the Android 14+ toggle writes the same global key is **inferred**, not confirmed.
- **Effect on speed (anecdotal):** one tester saw lower big-core clocks after toggling it, in the foreground
  ([#5035](https://github.com/termux/termux-app/issues/5035)). Unexplained, single report. Keep it the same for every
  session and record it.
- With Termux in the foreground (which §4.3 requires anyway), the CPU-use kill does not apply, and our session has far
  fewer than 32 processes. Turning the restriction off is insurance against an accidental app switch.

### 3.2 `termux-wake-lock`

- It makes the Termux service take a `PARTIAL_WAKE_LOCK` and a `WIFI_MODE_FULL_HIGH_PERF` Wi-Fi lock, and, if battery
  optimisation is on for Termux, opens the system dialog to turn it off
  ([TermuxService.java L303–L328](https://github.com/termux/termux-app/blob/master/app/src/main/java/com/termux/app/TermuxService.java#L303-L328)).
  Accept that dialog. The Termux notification then shows the lock as held; `termux-wake-unlock` releases it.
- **It keeps the CPU and Wi-Fi awake, but it does not keep Termux in the top-app cpuset**
  ([#5035](https://github.com/termux/termux-app/issues/5035)). It is necessary (SSH and the job survive the screen
  going off), not sufficient (§4.3).

### 3.3 Samsung battery settings

- **Per app:** Settings → Apps → Termux → Battery → **Unrestricted**; the same for Termux:API (One UI 5+ path per
  [9to5Google](https://9to5google.com/2023/02/17/galaxy-phone-unrestricted-access/), secondary).
- **Background usage limits:** Settings → Battery (older: Battery and device care → Battery) → Background usage limits →
  **Put unused apps to sleep** off, and add Termux and Termux:API to **Never sleeping apps**
  ([Samsung US](https://www.samsung.com/us/support/answer/ANS00088422/)). One user reports that choosing Unrestricted
  removes an app from all sleeping lists ([9to5Google](https://9to5google.com/2023/02/17/galaxy-phone-unrestricted-access/),
  forum quote; anecdotal).
- **Auto optimisation:** Settings → Device care → Auto optimisation → **Restart when needed** off. Samsung says it acts
  only *"when the device is not being used (Screen is off...)"* ([Samsung IN](https://www.samsung.com/in/support/mobile-devices/how-to-automatically-optimise-your-galaxy-device/)),
  so with the screen on it should not fire; off is cheap insurance against a reboot mid-session.
- **Power saving** off, **Adaptive power saving** off (§4.1).
- Adaptive battery / App Standby Buckets limit jobs, alarms and network for idle apps; an app with a visible activity or
  a foreground service is "active" ([Android docs](https://developer.android.com/topic/performance/appstandby)). Termux
  in the foreground with its service running is never idle, so this does not bite.

### 3.4 The screen

- Developer options → **Stay awake** ("screen will never sleep while charging",
  [dev options](https://developer.android.com/studio/debug/dev-options)) keeps the screen on, and so the lock screen off,
  for as long as the phone is plugged in. #27 requires the phone plugged in, so this covers the whole session.
- **Termux must be the app on screen**: not the launcher, not split-screen, not a pop-up view (§4.3). Leave the Termux
  terminal showing; the session runs over SSH, but the visible activity decides the cpuset.

### 3.5 Does a `nohup` job survive the SSH disconnect?

- Yes, from the mechanics: `nohup` ignores SIGHUP and `setsid` (util-linux, in the bootstrap) gives the job its own
  session, so closing the SSH connection does not signal it. The wizard's stage 6 already starts it as `nohup setsid
  python -u ... < /dev/null &`.
- What can still kill it is Android: the phantom process killer (§3.1), the Termux app being force-stopped or swiped
  away with no wake lock (which takes `sshd` too), or a reboot (§3.3). The runner resumes after any of these
  ("run the same command again"), but a killed run restarts from the start.
- Not tested on the M34 (**unconfirmed in practice**). A short dry run (§8, step 16) confirms it in five minutes.

---

## 4. What distorts CPU latency on this phone

### 4.0 The phone's software

- The M34 shipped with Android 13 and a promise of 4 OS upgrades
  ([GSMArena](https://www.gsmarena.com/samsung_galaxy_m34_announced_with_4_os_updates_50mp_camera_with_ois-news-59116.php)).
  One UI 7 (Android 15) reached India in May 2025 ([SamMobile](https://www.sammobile.com/2025/05/27/galaxy-m34-one-ui-7-india/)).
  One UI 8 (Android 16) followed in October 2025, firmware M346BXXU8EYI8
  ([SamMobile](https://www.sammobile.com/news/samsung-releases-one-ui-8-android-16-update-galaxy-m34)).
- A One UI 9 (Android 17) beta has run in India since September 2026
  ([SamMobile](https://www.sammobile.com/news/galaxy-m34-one-ui-9-beta-update-released/)).
- **So a stock M34 is most likely on One UI 8 / Android 16.** Do not install a One UI 9 beta or any update during the
  study. Record the build: `getprop ro.build.display.id` (the bench already records `ro.build.version.release`).

### 4.1 Samsung settings that change the CPU

| Setting | What it does | On the M34? | Set to |
|---|---|---|---|
| **Performance profile** (Standard / Light), Settings → Device care → Performance profile | Light *"slightly reduces its maximum processing speed"* ([Samsung CA](https://www.samsung.com/ca/support/mobile-devices/light-performance-profile-on-your-samsung-galaxy)); Standard is the default ([Samsung US](https://www.samsung.com/us/support/answer/ANS10001953)) | Samsung's 2023 page limits Light to Z and S models from the S23 ([Samsung IN](https://www.samsung.com/in/support/mobile-devices/how-to-use-galaxy-s23-light-mode/)). **Unconfirmed; likely absent** | Standard, if present |
| **Processing speed** (Optimized / High / Maximum), Device care | Raises CPU limits for non-game apps ([MakeUseOf](https://www.makeuseof.com/settings-that-actually-make-your-samsung-phone-faster/), secondary) | **Unconfirmed**; no official Samsung page found | Leave at the default (Optimized); record it |
| **Power saving**, Settings → Battery → Power saving | Includes *"Limit CPU speed to 70%"* ([Samsung US](https://www.samsung.com/us/support/answer/ANS10001953)) | yes | **Off**, and Adaptive power saving off |
| **Battery protection**, Settings → Battery → Battery protection (One UI 6.1+) | Basic: stop at 100%, resume at 95%. Adaptive: Maximum while asleep. Maximum: stop at 80% ([Samsung UK](https://www.samsung.com/uk/support/mobile-devices/galaxy-battery-protection-feature-in-one-ui-6-1)) | yes (One UI 8) | One fixed choice for every session; record it |
| **Fast charging**, Settings → Battery → Charging settings | Heat while under load | yes | **Off** ([Guiding Tech](https://www.guidingtech.com/fix-samsung-one-ui-battery-drain), secondary) |
| **Game Booster / GOS** | *"GOS does not manage the performance of non-gaming apps"* ([XDA](https://www.xda-developers.com/samsung-throttling-popular-android-apps/)) | yes | Do **not** add Termux as a game |
| **Pause USB Power Delivery** (bypass charging) | Games only; supported list includes the A53 and M33 but **not the M34** ([Samsung UK](https://www.samsung.com/uk/support/mobile-devices/what-is-the-pause-usb-power-delivery-feature)) | no | n/a |

### 4.2 Heat and charging

- Samsung lists *"using demanding apps while fast charging"* as a heat cause, and limits *"performance speed"* and stops
  charging on heat warnings ([Samsung UK](https://www.samsung.com/uk/support/mobile-devices/what-to-do-when-your-samsung-galaxy-device-heats-up-while-in-use/),
  [Samsung UK](https://www.samsung.com/uk/support/mobile-devices/what-to-do-when-your-phone-heats-up/)). **No thresholds are
  published.** SSRM, Samsung's thermal manager, has no primary documentation I could find (**unconfirmed**).
- UL recommends unplugging during benchmarks because *"the battery charging circuitry also generates heat"*
  ([UL](https://support.benchmarks.ul.com/support/solutions/articles/44001791209-my-benchmark-score-seems-to-vary-wildly-between-runs)).
  MLPerf Mobile runs on battery after a full charge, at 20–25 °C room temperature, with no flush contact with surfaces
  ([MLPerf Mobile §6.1](https://ar5iv.labs.arxiv.org/html/2012.02328)). Our protocol keeps the phone plugged in (#27). So:
  fast charging off, start each session near full or at the battery-protection cap, so the charge current during runs is
  small and similar between runs (**inferred**). Whether the phone runs from the adapter while held at the cap is
  **unconfirmed**.
- The phone has no throttle signal Termux can read (§4.5), and the runner already records "unknown". Its proxies are
  the battery temperature at each gate and, from the laptop over adb, `dumpsys thermalservice` (thermal status and HAL
  temperatures; the `shell` user holds `DUMP`: [Shell manifest](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/packages/Shell/AndroidManifest.xml)).
- The 3-minute wait is shorter than published cool-downs: MLPerf Mobile uses 0–5 min between tests and 10 min between
  suite runs; UL suggests 15 min. Our protocol is fixed, so log the start temperature and compare it across cells.

### 4.3 big.LITTLE and where Termux's threads run (the biggest risk)

- **On One UI 8.x, a backgrounded Termux loses the big cores.** An S26 Ultra (One UI 8.5) showed Termux in cpuset
  `/top-app`, mask `ff`, when focused. Within about 1 s of switching away it moved to Samsung's `/moderate` cpuset, mask
  `33` (cpu0-1, cpu4-5), with the big cores idling at 883 MHz. *"wake-lock keeps the CPU awake but doesn't keep the app
  in `/top-app`"* ([#5035](https://github.com/termux/termux-app/issues/5035)).
  - Every process of the Termux uid moves together. `taskset -cp 0-7` is clamped to the mask, `taskset -cp 6,7` gives
    EINVAL, a priority notification changes nothing, and writing to `/dev/cpuset/top-app/tasks` over adb is denied
    ([#5086](https://github.com/termux/termux-app/issues/5086)).
  - A stock Pixel 10 Pro moved it to `/foreground` (7 cores) instead ([#5086](https://github.com/termux/termux-app/issues/5086)).
  - Reported on S24+, S25 Ultra, S26 Ultra and Tab S9 Ultra on One UI 8.0/8.5; the M34 was not tested (**likely, unconfirmed
    for the M34**). On One UI 7 the reporters saw all cores used.
- **Our session runs under sshd, a child of the Termux app, so it is in the same uid and the same cgroup.** Children
  inherit cgroups ([cgroups(7)](https://man7.org/linux/man-pages/man7/cgroups.7.html)), and AOSP sets the app's process
  group on its phantom processes too (`setProcessGroup` in
  [PhantomProcessList.java](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/services/core/java/com/android/server/am/PhantomProcessList.java)).
  **Driving the run over SSH does not escape this.** The Termux app itself must be the visible activity, the screen on
  and unlocked, for the whole session.
- AOSP also drops the top app to the background group once the device sleeps ("top-sleeping") in
  [OomAdjuster.java](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/services/core/java/com/android/server/am/OomAdjuster.java).
  So the screen must stay on, not just the wake lock.
- `os.sched_setaffinity` is allowed for apps ([bionic SYSCALLS.TXT](https://android.googlesource.com/platform/bionic/+/refs/heads/main/libc/SYSCALLS.TXT),
  not in the [app seccomp blocklist](https://android.googlesource.com/platform/bionic/+/refs/heads/main/libc/SECCOMP_BLOCKLIST_APP.TXT)),
  but it is clamped by the cpuset, and a cpuset move resets it ([kernel cpusets doc](https://docs.kernel.org/admin-guide/cgroup-v1/cpusets.html)).
  We do not pin anyway (the protocol does not).
- **Core numbering (verify on the phone).** The Exynos 1280 has 2× A78 at 2.4 GHz and 6× A55 at 2.0 GHz
  ([Samsung](https://semiconductor.samsung.com/processor/mobile-processor/exynos-1280/)). That cpu0–5 are the A55s and
  cpu6–7 the A78s is **unconfirmed**. `grep 'CPU part' /proc/cpuinfo` gives `0xd05` for A55 and `0xd41` for A78
  ([Linux cputype.h](https://github.com/torvalds/linux/blob/master/arch/arm64/include/asm/cputype.h)), and the bench's
  `cpu_max_mhz` list already shows the clusters.
- **How 4 ORT threads land on 2 big + 6 little.** With an explicit thread count ORT sets *"no affinity"*, so the
  scheduler places the threads ([ORT threading doc](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)).
  At best 2 threads run on A78s and 2 on A55s. One A55 is roughly 3× slower than one A76/A78-class core
  (NCNN's RK3588 table, 603 vs 187 ms for ResNet-50 single-thread: [ncnn benchmark](https://github.com/Tencent/ncnn/blob/master/benchmark/README.md)).
  - ORT does detect a mixed-core CPU (`is_hybrid_ = cpuinfo_get_uarchs_count() > 1`:
    [cpuid_info.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/common/cpuid_info.cc)) and then
    splits work into 4× more blocks (`DegreeOfParallelism`:
    [threadpool.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/common/threadpool.cc)), claimed
    dynamically, so the fast threads take more. But MLAS FP32 convolution slices each conv equally per thread
    ([convolve.cpp](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/convolve.cpp)), so the
    A55 slices gate those ops (from the source; **the net effect on the M34 is unmeasured**).
  - On an RK3588S, NCNN ran ResNet-50 in 50.8 ms on all 8 cores and 27.5 ms on the 4 big ones
    ([ncnn benchmark](https://github.com/Tencent/ncnn/blob/master/benchmark/README.md)). XNNPACK maintainers report 2–3×
    slowdowns on many phones when the thread count exceeds the performant cores
    ([executorch #8763](https://github.com/pytorch/executorch/issues/8763)).
  - **What that means for us:** 4 threads on this SoC are probably not faster than 2, and they are more sensitive to
    placement. The phone's latency has more run-to-run variance than a Pi's. The protocol keeps 4 threads (it is the
    cross-platform rule), so record where the threads ran (§4.5) rather than change it.
- **Spinning.** `session.intra_op.allow_spinning` defaults to on ([config keys](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)):
  the 3 pool threads keep spinning after each `Run()`, while Python does pre and post (**inferred**). That costs heat and
  can slow `pre`/`post` slightly when the spinning threads share a core with the main thread. It is the same on every
  platform, so it is consistent; record the default.

### 4.4 Other load

- Keep Developer options → **Stay awake** on: it keeps the screen on while plugged in
  ([dev options](https://developer.android.com/studio/debug/dev-options)). Fixed low brightness, adaptive brightness off,
  Standard (60 Hz) refresh rate (Samsung's heat advice; exact path **unconfirmed**: likely Settings → Display → Motion
  smoothness). Do Not Disturb on, other apps closed.
- Each `termux-battery-status` call goes through `am broadcast` to the Termux:API app
  ([termux-api](https://github.com/termux/termux-api)): it is only called at the gate, outside the timed images. Fine.
- Terminal output is rendered by the Termux app (**inferred**). The runner writes to files under nohup, so this is small.

### 4.5 What a non-root Termux user can read

Termux from F-Droid targets SDK 28 ([gradle.properties](https://github.com/termux/termux-app/blob/master/gradle.properties)),
so it runs in the `untrusted_app_27` SELinux domain ([seapp_contexts](https://android.googlesource.com/platform/system/sepolicy/+/refs/heads/main/private/seapp_contexts)),
which shares the `/proc` and `/sys` rules of every untrusted app.

| What | From Termux | Over adb from the laptop | Source |
|---|---|---|---|
| `cpu*/cpufreq/scaling_cur_freq`, `scaling_max_freq`, `scaling_governor`, `cpuinfo_max_freq`, `stats/time_in_state`, `policy*/related_cpus` | **yes** (label `sysfs_devices_system_cpu`, readable by every domain unless the vendor relabels); read from Termux on an S26 Ultra. **M34 unconfirmed** | yes | [genfs_contexts](https://android.googlesource.com/platform/system/sepolicy/+/refs/heads/main/private/genfs_contexts), [domain.te](https://android.googlesource.com/platform/system/sepolicy/+/refs/heads/main/private/domain.te), [#5035](https://github.com/termux/termux-app/issues/5035) |
| `cpuinfo_cur_freq` | no (mode 0400) | no | [cpufreq.c](https://github.com/torvalds/linux/blob/master/drivers/cpufreq/cpufreq.c) |
| `/sys/class/thermal/thermal_zone*/temp` | **probably no** (`sysfs_thermal` is for system domains). **Unconfirmed on the M34**: test once | unclear | [app_neverallows.te](https://android.googlesource.com/platform/system/sepolicy/+/refs/heads/main/private/app_neverallows.te) |
| `/sys/class/power_supply/battery/temp` | **no** (`sysfs_batteryinfo`); use `termux-battery-status` | yes | [domain.te](https://android.googlesource.com/platform/system/sepolicy/+/refs/heads/main/private/domain.te), [shell.te](https://android.googlesource.com/platform/system/sepolicy/+/refs/heads/main/private/shell.te) |
| `termux-battery-status` | **yes**, with the Termux:API app: `health`, `percentage`, `plugged`, `status`, `temperature` (°C, from `EXTRA_TEMPERATURE`/10), `current` (µA) | — | [BatteryStatusAPI.java](https://github.com/termux/termux-api/blob/master/app/src/main/java/com/termux/api/apis/BatteryStatusAPI.java) |
| `/proc/stat`, `/proc/loadavg` | **no** (closed to apps since Android 8) | yes | [untrusted_app_all.te](https://android.googlesource.com/platform/system/sepolicy/+/refs/heads/main/private/untrusted_app_all.te) |
| `/proc/cpuinfo` | yes | yes | [domain.te](https://android.googlesource.com/platform/system/sepolicy/+/refs/heads/main/private/domain.te) |
| `/dev/cpuset/*/cpus` | no | **yes** | [app_neverallows.te](https://android.googlesource.com/platform/system/sepolicy/+/refs/heads/main/private/app_neverallows.te) |
| `/proc/self/cpuset`, `Cpus_allowed_list` in `/proc/self/status`, `/proc/self/task/*/stat` (field 39 = last CPU) | **yes** (used from Termux in [#5035](https://github.com/termux/termux-app/issues/5035)/[#5086](https://github.com/termux/termux-app/issues/5086)) | yes | [proc_pid_stat(5)](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html) |
| `dumpsys thermalservice` (thermal status, HAL temperatures, headroom) | no (needs `DUMP`) | **yes** | [ThermalManagerService.java](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/services/core/java/com/android/server/power/ThermalManagerService.java) |

What the code does today, given this table:
- `studybench.cpu_temp_c()` and the runner's `cpu_temp_c()` read thermal zones: probably `None` on the phone (the
  runner already says "the phone shows no CPU temperature").
- `studysession.battery_temp_c()` tries `/sys/class/power_supply/battery/temp` first (denied), then
  `termux-battery-status` with a 10 s timeout: that works once the Termux:API **app** is installed.
- `studybench.cpu_mhz()` reads **cpu0's** `scaling_cur_freq`. If cpu0 is an A55, the `cpu_mhz` column shows the little
  cluster, not the A78s that carry most of the work (§7, candidate).

---

## 5. SSH into Termux

The Termux wiki's Remote Access page was behind a bot wall (Anubis) on 2026-10-25. Everything below comes from the
`openssh` package source in termux-packages instead, which is what actually runs.

**sshd.**
- `pkg install openssh`, then run `sshd`. It daemonises and listens on **8022**: Termux patches the compiled-in
  default (`options->ports[...] = 8022 /* SSH_DEFAULT_PORT */` in
  [servconf.c.patch](https://github.com/termux/termux-packages/blob/master/packages/openssh/servconf.c.patch)). Android
  does not let apps bind ports below 1024.
- It does **not** start by itself after a reboot or after Termux is killed. Run `sshd` again. (`termux-services` can
  supervise it with `sv-enable sshd`; the package ships `sshd -D -e` as the service:
  [sv/sshd.run.in](https://github.com/termux/termux-packages/blob/master/packages/openssh/sv/sshd.run.in). Not needed for
  one session.)
- Host keys are made at install. **The laptop has the Play build's old host key for the phone's address**, so the first
  connection after reinstalling will fail with *REMOTE HOST IDENTIFICATION HAS CHANGED*. Fix it with
  `ssh-keygen -R "[<phone IP>]:8022"` on the laptop.

**User name and password.**
- **Any user name works.** Termux patches `auth.c`: *"Effectively a single-user system, use current user no matter
  supplied user"* (`pw = getpwuid(getuid())`, [auth.c.patch](https://github.com/termux/termux-packages/blob/master/packages/openssh/auth.c.patch)).
  The real user is `u0_aNNN` (`whoami`). For the wizard, any value of `PI_USER` works; use `whoami`'s output to keep logs
  honest.
- The password is set with `passwd` (from `termux-auth`) and checked by `termux_auth`
  ([auth-passwd.c.patch](https://github.com/termux/termux-packages/blob/master/packages/openssh/auth-passwd.c.patch)).
  The wizard's stage 2 needs it once, for `ssh-copy-id`.
- Keys go in `~/.ssh/authorized_keys`. The package's postinst creates `~/.ssh` (mode 700) and that file
  ([build.sh](https://github.com/termux/termux-packages/blob/master/packages/openssh/build.sh)).

**The laptop's `~/.ssh/config`** (Git Bash's `~` is `C:\Users\srikr`). `ssh-copy-id`, `ssh` and `tar | ssh` all read it:

```
Host m34
    HostName <the phone's IP>
    Port 8022
    User u0_aNNN
    ServerAliveInterval 30
Host m34usb
    HostName 127.0.0.1
    Port 8022
    User u0_aNNN
```

The wizard then takes `m34` as the host. `PI_USER` can be the same `u0_aNNN`, since the wizard puts `user@host` in front.

**Finding the phone's IP.**
- In Termux: `ifconfig wlan0` (`net-tools`, in the bootstrap). Termux targets SDK 28, so Android 11's netlink limit for
  apps targeting SDK 30+ should not apply (**inferred**). `ip` needs `pkg install iproute2`.
- In Settings: Connections → Wi-Fi → the gear by the network → IP address (exact One UI 8 wording **unconfirmed**).
- Make the address stable for the session: either a DHCP reservation on the router or set it again in `~/.ssh/config`
  each time.

**Wi-Fi client isolation.** Campus and guest networks often block traffic between clients. Test from the laptop:
`ssh -o ConnectTimeout=5 m34 true`. If it times out while the phone's IP is right:
1. Use the **laptop's Mobile hotspot** (Windows Settings → Network & internet → Mobile hotspot) and join it from the
   phone. The phone stays a Wi-Fi client, so its radio load does not change. The phone's IP is then usually in
   192.168.137.0/24 (the Windows ICS default; **unconfirmed** for this laptop).
2. Or USB: on the phone, Developer options → USB debugging on; on the laptop (Android platform-tools)
   `adb forward tcp:8022 tcp:8022`, then `ssh m34usb` ([adb docs, "forward"](https://developer.android.com/tools/adb)).
   The forward lasts until the cable or the adb server goes away, so a dropped cable drops SSH (the session goes on).
   The USB cable also powers the phone from the laptop's port instead of the wall charger: record the power source.
3. Avoid the phone's own hotspot: it adds radio load and heat on the device under test.

**The tools the wizard uses on the remote side** (official `Contents-aarch64` index of 2026-10-25, matched against
the bootstrap list in [generate-bootstraps.sh](https://github.com/termux/termux-packages/blob/master/scripts/generate-bootstraps.sh)):

| Tool | Package | In a fresh bootstrap? |
|---|---|---|
| `bash` | bash | yes |
| `sh` (used by `ssh-copy-id` on the remote) | dash | yes |
| `tar` | tar | yes |
| `find`, `xargs` (`-print0`, `-0`) | findutils (GNU) | yes |
| `sort -z`, `sha256sum`, `cut`, `nohup`, `tr`, `cat`, `mkdir`, `tail`, `timeout` | coreutils (GNU) | yes |
| `sed -E` | sed (GNU) | yes |
| `awk` | gawk | yes |
| `setsid` | util-linux | yes |
| `df` | **termux-tools: a wrapper around Android's `/system/bin/df`** (toybox), not coreutils | yes. Toybox `df` takes `-P` and `-k`, so `df -Pk "$HOME"` should give the POSIX columns (**inferred**; check once) |
| `getprop` (the bench's Android facts) | termux-tools: a wrapper around the system `getprop` | yes |
| `kill -0` | bash builtin (coreutils also ships `kill`) | yes |
| `/proc/$pid/cmdline` of our own runner | kernel; same uid, so readable | yes |
| `ifconfig` | net-tools | yes |
| `python` | python | **no**: `pkg install python` |
| `sshd`, `ssh-copy-id` | openssh | **no**: `pkg install openssh` |
| `passwd` | termux-auth | pulled in by openssh |
| `termux-wake-lock`, `termux-change-repo` | termux-tools | yes |
| `termux-battery-status` | termux-api | **no**: `pkg install termux-api`, plus the Termux:API app |
| `ip` | iproute2 | no (not needed) |

What does not exist on the phone and the wizard reads anyway: `/proc/device-tree/model` (stage 2 prints an empty model;
harmless, its Pi check is guarded by `PLATFORM == pi5`), `vcgencmd` and the fan file (stage 3, guarded by `pi*`).

**The venv.** `python -m venv --system-site-packages --without-pip ~/sb` makes `~/sb/bin/python`, which the wizard
expects. It sees numpy, onnxruntime and cv2 from `$PREFIX/lib/python3.14/site-packages`. `--without-pip` skips
`ensurepip`, which the venv does not need.

---

## 6. Expected speed

**No published YOLO latency exists for any Exynos 1280 phone** (A53, A33, M33, M34, A25, F34) in any framework (searched
papers, Ultralytics docs, ai-benchmark, ORT issues). Everything below is scaled, and is an estimate.

Anchors:

| Model | Input | Runtime, precision | Threads | Device | ms per image | Source |
|---|---|---|---|---|---|---|
| YOLO11s | 640 | ONNX (Ultralytics 8.4.14), FP32 | ORT default | Pi 5, 4× A76 at 2.4 GHz | **366.83** (was 449.85 with 8.3.152) | [Ultralytics Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/), Feb 2026; [older table](https://github.com/ultralytics/ultralytics/blob/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/docs/en/guides/raspberry-pi.md) |
| YOLO26s | 640 | ONNX / NCNN, FP32 | default | Pi 5 | 355.69 / 172.88 | same page, Jul 2026 |
| YOLO11n | 320 | TFLite FP32 / full-int8 | 4 | Galaxy A34 (Dimensity 1080: 2× A78 at 2.6 + 6× A55 at 2.0, the nearest twin) | 80 / 74.5, with camera preprocessing | [Zenodo 20671391](https://zenodo.org/records/20671391), Jun 2026 |
| ResNet-50 | 224 | ORT CPU EP, FP32 → INT8 QDQ | fixed | Pi 5 / Orin (A78AE) / i.MX8M (A53, no dot product) | 1.83× faster / 2.11× faster / **1.65× slower** | [arXiv 2609.16085](https://arxiv.org/pdf/2609.16085v1), Sep 2026 preprint |

Cross-chip scale: Geekbench 6 gives the M34 956 single-core / 2,032 multi-core
([GSMArena, 2023](https://www.gsmarena.com/samsung_galaxy_f34_support_page_goes_livegalaxy_m34_pops_up_on_geekbench-news-59018.php))
against the Pi 5's 833–904 / 1,805–2,164 ([Geerling sbc-reviews #21](https://github.com/geerlingguy/sbc-reviews/issues/21)).
So one A78 ≈ 1.06–1.15× one A76 at the same clock, but the M34 has only two of them.

**INT8 on this SoC.** MLAS uses the dot-product (SDOT/UDOT) INT8 kernels only when the CPU reports them
([platform.cpp](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/platform.cpp)). The A78 has
them; on the A55 they are optional per Arm. **Check on the phone:** `grep -m1 -o asimddp /proc/cpuinfo`. The kernel shows
a feature only if every core has it. With dot product, expect INT8 1.3–1.9× faster than FP32. Without it, INT8 can be
slower than FP32 (the i.MX8M row).

**Estimate for YOLO11s at 640, ORT 1.31, 4 threads, unpinned, Termux in the foreground:**
- FP32 inference **600–1,000 ms** per image (plan 800); INT8 **350–750 ms** (plan 500).
- Pre and post on the phone, about **10–30 ms** (no source; estimate).
- In the background on One UI 8 (little cores only): expect **2–3× slower**. A number in that range means the run is
  invalid, not slow.

**Session lengths** (gate 3 min before each run; the speed test is 4 runs of 200 images: 2 reference FP32 runs, one
FP32 cell, one INT8 cell; the full cell is 3 runs: 2 references of 200 and 1,610 images):

| Plan | Images | Inference | Gates | Total |
|---|---|---|---|---|
| `m34-speedtest` | 600 FP32 + 200 INT8 | about 7.5–13 min | 12 min | **about 20–30 min**, plus loads and warm-ups (about 1 min) |
| `m34-fullcell` | 2,010 FP32 | about 21–35 min | 9 min | **about 30–45 min** |

The plan's own note (25–40 min for the speed test) is consistent. Budget the full cell at 45 min and stop it if its
first 200 images run at more than about 2 s each (§4.3).

---

## 7. Candidate repo changes (not made; for the owner to decide, each a possible sub-issue of #1)

**Wizard (`study/pi/wizard.sh`), for the phone:**
1. **Stage 5 installs with `pip install` of `PI_PINS`**, which cannot work on the phone (no Android wheels on PyPI,
   §2). A phone branch should instead check for, or run, `pkg install x11-repo && pkg install python python-numpy
   python-onnxruntime opencv-python termux-api openssh`, then `python -m venv --system-site-packages --without-pip ~/sb`.
2. **Stage 1's "have ready" list and stage 3** are Pi-specific (27 W supply, fan, SD card). The phone needs its own: F-Droid
   Termux + Termux:API, `sshd` running, Termux on screen, Stay awake, child process restrictions off, charger.
3. **Put the #27 checks Termux cannot read to the user and record the answers** (#27's third acceptance criterion):
   Termux visible on screen, Stay awake on, "Disable child process restrictions" on, power saving off, fast charging off,
   battery-protection mode, performance profile / processing speed, charger and cable, room temperature. The Pi's
   `ask_room` is skipped for `m34` today (`[[ "$PLATFORM" == pi* ]] || return 0`); the phone needs the room temperature
   at least as much.
4. **Tell the user, at stage 6, to bring Termux to the front and leave it there**, and say why (§4.3). This is the step
   most likely to be missed when driving from the laptop.

**Runner (`study/scripts/studysession.py`), for `m34`:**
5. `platform_checks()` returns no checks for `m34`. Readable from Termux (§4.5), so checkable:
   - battery ≥ 50 % and plugged (`termux-battery-status`: `percentage`, `plugged`);
   - the Termux:API app answers (the 10 s timeout already guards the hang);
   - **the runner's own cpuset is `/top-app`** (`/proc/self/cpuset`) and `Cpus_allowed_list` covers all 8 CPUs. Read at
     every gate, this catches a backgrounded Termux. A gate that finds another cpuset could refuse to start the run.
   - `asimddp` in `/proc/cpuinfo` (records whether INT8 can use the dot-product kernels).
6. **Record more of the battery at each gate**, not only `temperature`: `percentage`, `plugged`, `status`, `current`
   ([BatteryStatusAPI.java](https://github.com/termux/termux-api/blob/master/app/src/main/java/com/termux/api/apis/BatteryStatusAPI.java)).
   That gives #27's "plugged in, battery above 50%" as data.

**Bench (`study/scripts/studybench.py`), for the phone:**
7. `cpu_mhz()` reads **cpu0** only. On the M34 cpu0 is probably an A55. Log the big cluster too (the CPU whose
   `cpuinfo_max_freq` is highest, or every `policy*/scaling_cur_freq`), and its `scaling_max_freq`, which may show a thermal
   cap (**unconfirmed** that Samsung caps through it).
8. Record, at the start, every 50 images and at the end: `/proc/self/cpuset` and, per ORT thread, the last CPU
   (`/proc/self/task/*/stat` field 39) and its CPU time (fields 14–15). That shows how many of the 4 threads ran on the
   A78s, which is the main unknown of every phone number (§4.3).
9. `time_in_state` per cluster before and after each run (cheap; shows the time spent at each clock).

**PLATFORMS.md, § Galaxy M34:**
10. Add: F-Droid builds of Termux and Termux:API (both from there), `pkg upgrade` once and then freeze, `termux-change-repo`
    to the default mirror, the download size (about 190 MB, 1 GiB installed), `pkg install termux-api openssh`, and the
    phone setup of §8.
11. Its install line omits `termux-api` and `openssh`, which the session needs.
12. Note that the phone's throttle state is "unknown" and that its latency depends on Termux being in the foreground; and
    that the ±3 % drift check may trip more often on a 2+6 big.LITTLE phone than on a Pi (thread placement varies run
    to run; **inferred**).

---

## 8. Setup checklist

### On the phone, once

1. Uninstall **every** Play Store Termux app and plugin (Termux, and Termux:API if any). Back up nothing: there is nothing
   of the study on it.
2. From F-Droid (the app, or the APK from the website): **Termux 0.118.3** and **Termux:API** (same source). Open Termux
   once and let the bootstrap finish.
3. In Termux, pin the default mirror: `termux-change-repo` → *Single mirror* → the default. Then:

   ```bash
   pkg upgrade -y
   pkg install -y x11-repo
   pkg install -y python python-numpy python-onnxruntime opencv-python termux-api openssh
   python -m venv --system-site-packages --without-pip ~/sb
   ~/sb/bin/python -c "import numpy, cv2, onnxruntime as o; print(numpy.__version__, cv2.__version__, o.__version__, o.get_available_providers())"
   ```
   Expect `2.5.4 4.14.0 1.31.0` plus a harmless `Unsupported platform (android)` warning (§2). About 190 MB to download.
   **No further `pkg upgrade` until the study's phone sessions are done.**
4. `termux-battery-status` must print JSON with `temperature` (if it hangs, the Termux:API app is missing or from another
   source). Allow its permission prompt if one appears.
5. Record the facts once: `getprop ro.build.display.id`, `grep -E 'processor|CPU part' /proc/cpuinfo`,
   `grep -m1 -o asimddp /proc/cpuinfo`, `cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq`,
   `cat /sys/class/thermal/thermal_zone0/temp` (expect "Permission denied"; if it reads, note it).
6. SSH: `passwd` (choose a password), `whoami` (note `u0_aNNN`), `sshd`, `ifconfig wlan0` (note the IP).
7. Settings:
   - About phone → Software information → tap *Build number* 7 times.
   - Developer options → **Disable child process restrictions: on**; **Stay awake: on**. (USB debugging on only if you
     will use the adb fallback.)
   - Apps → Termux → Battery → **Unrestricted**; the same for Termux:API.
   - Battery → Background usage limits → **Put unused apps to sleep: off**; add Termux and Termux:API to **Never sleeping apps**.
   - Battery → **Power saving: off** (and Adaptive power saving off).
   - Battery → Charging settings → **Fast charging: off**. Battery → Battery protection → one fixed choice; note it.
   - Device care → **Auto optimisation → Restart when needed: off**.
   - Device care → Performance profile, if present: **Standard**. Processing speed, if present: leave the default; note
     which of the two exists.
   - Display: fixed low brightness, adaptive brightness off, refresh rate Standard. Do Not Disturb on.
   - Do **not** add Termux to Game Booster. Do **not** install the One UI 9 beta or any update during the study.
8. Charger: the same wall charger and cable every session. The phone flat on a hard surface, not a cushion, out of
   the sun, away from the laptop's exhaust.

### On the phone, before every session

9. Charge to above 50 % (ideally near full or the protection cap). Plugged in.
10. Close other apps. Open Termux, run `termux-wake-lock` (accept the battery-optimisation dialog the first time) and
    `sshd` (if not already running: `pgrep sshd`).
11. **Leave Termux open on screen, unlocked, for the whole session.** Do not switch apps, do not open the notification
    shade for long, do not let the phone lock. This is what keeps the runner on the big cores (§4.3).

### On the laptop (Git Bash)

12. `ssh-keygen -R "[<phone IP>]:8022"` (drops the old Play build's host key).
13. Add the `m34` (and, if needed, `m34usb`) block of §5 to `~/.ssh/config`.
14. `ssh -o ConnectTimeout=5 m34 'uname -m; whoami; df -Pk "$HOME" | tail -1'`. If it times out: client isolation (§5);
    use the laptop hotspot, or `adb forward tcp:8022 tcp:8022` and `ssh m34usb`.
15. Optional, once (USB debugging on): `adb shell settings get global settings_enable_monitor_phantom_procs` (expect
    `false` after step 7), `adb shell cat /dev/cpuset/top-app/cpus /dev/cpuset/foreground/cpus /dev/cpuset/background/cpus`
    and `adb shell dumpsys thermalservice | head -40`. Then unplug USB so the phone charges from its wall charger.
16. **Dry run first** (about 5 min): the wizard with `WIZ_DRY_RUN=1` on `plans/m34-speedtest.json`. Disconnect SSH in
    the middle of it, reconnect, and check it is still running (§3.5). While it runs, `ssh m34 'cat /proc/$(cat
    ~/uav/study/bench/m34/m34-speedtest/runner.pid)/cpuset'` should say `/top-app`. Switch Termux to the background on
    the phone for a few seconds and run it again: if it now says something else, §4.3 applies to the M34, and the
    rule "Termux on screen" is not optional. Then delete the dry run's output, which sits where the real session will
    write and would otherwise be resumed as complete: `ssh m34 'rm -rf ~/uav/study/bench/m34/m34-speedtest'`.
17. Run the speed test, then the full cell (§6 for their length). Record per session: the room temperature, the
    settings of step 7, the charger, the battery mode, and the One UI build.

The wizard's stage 5 tries `pip install` on a missing `~/sb`; with steps 3 done by hand, it finds the venv and skips
the install (§7, item 1).

---

## 9. Not confirmed

- That the M34 (One UI 8) moves a backgrounded Termux to a little-core cpuset. Shown on S24+/S25 Ultra/S26 Ultra/Tab S9
  Ultra only; step 16 tests it.
- The M34's core numbering (cpu0–5 A55, cpu6–7 A78), its cpuset masks, and whether its A55s have dot product (`asimddp`).
- Whether `scaling_cur_freq` and thermal zones are readable from Termux on the M34, and whether Samsung's thermal caps
  show in `scaling_max_freq`.
- Whether the M34 has the Performance profile and/or Processing speed menus, and what Processing speed does.
- SSRM's behaviour and any temperature thresholds; whether the phone runs from the adapter at the battery-protection cap.
- That the Android 14+ "Disable child process restrictions" toggle writes `settings_enable_monitor_phantom_procs`.
- Why the Play build reported `NO_PUBKEY 5A897D96E57CF20C` (inferred: its own keyring).
- That a stale mirror passes `pkg`'s check (inferred from the code, not observed).
- Toybox `df -Pk` output on the M34; `ifconfig` working under Termux on Android 16.
- The pre/post time on the phone, and every latency in §6: no Exynos 1280 YOLO number is published.
- The Termux wiki (wiki.termux.com) could not be read (bot wall); the SSH facts come from the package source instead.
- Whether ORT 1.31 has the `session.intra_op.spin_duration_us` key; whether ORT's cpuinfo reports the M34 as hybrid.

---

## Sources

Each claim carries its URL inline. The main primary sources:

- Termux app: [README](https://github.com/termux/termux-app), [releases](https://github.com/termux/termux-app/releases),
  [F-Droid page](https://f-droid.org/en/packages/com.termux/), [TermuxService.java](https://github.com/termux/termux-app/blob/master/app/src/main/java/com/termux/app/TermuxService.java),
  [gradle.properties](https://github.com/termux/termux-app/blob/master/gradle.properties) (targetSdk 28),
  [discussion #4000](https://github.com/termux/termux-app/discussions/4000), issues
  [#5035](https://github.com/termux/termux-app/issues/5035) and [#5086](https://github.com/termux/termux-app/issues/5086).
- Play build: [termux-play-store org](https://github.com/termux-play-store), [its termux-packages fork](https://github.com/termux-play-store/termux-packages).
- Packages: termux-packages build files for [onnxruntime](https://github.com/termux/termux-packages/blob/master/packages/onnxruntime/build.sh),
  [opencv](https://github.com/termux/termux-packages/blob/master/x11-packages/opencv/build.sh),
  [qt6-qt5compat](https://github.com/termux/termux-packages/blob/master/x11-packages/qt6-qt5compat/build.sh),
  [openssh](https://github.com/termux/termux-packages/tree/master/packages/openssh),
  [termux-tools](https://github.com/termux/termux-packages/blob/master/packages/termux-tools/build.sh),
  [generate-bootstraps.sh](https://github.com/termux/termux-packages/blob/master/scripts/generate-bootstraps.sh);
  the official indexes at `packages.termux.dev/apt/termux-main` and `.../termux-x11` (fetched 2026-10-25); issues
  [#31634](https://github.com/termux/termux-packages/issues/31634), [#28769](https://github.com/termux/termux-packages/issues/28769),
  [#29145](https://github.com/termux/termux-packages/issues/29145), [#30428](https://github.com/termux/termux-packages/issues/30428),
  [#22822](https://github.com/termux/termux-packages/issues/22822), [#31513](https://github.com/termux/termux-packages/issues/31513).
- termux-tools v1.45.0: [pkg.in](https://github.com/termux/termux-tools/blob/v1.45.0/scripts/pkg.in), [Makefile.am](https://github.com/termux/termux-tools/blob/v1.45.0/scripts/Makefile.am) (the `df`/`getprop` wrappers).
- Termux:API: [BatteryStatusAPI.java](https://github.com/termux/termux-api/blob/master/app/src/main/java/com/termux/api/apis/BatteryStatusAPI.java).
- Android: [phantom processes (agnostic-apollo)](https://github.com/agnostic-apollo/Android-Docs/blob/master/en/docs/apps/processes/phantom-cached-and-empty-processes.md),
  AOSP [sepolicy](https://android.googlesource.com/platform/system/sepolicy/+/refs/heads/main/private/),
  [OomAdjuster.java](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/services/core/java/com/android/server/am/OomAdjuster.java),
  [PhantomProcessList.java](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/services/core/java/com/android/server/am/PhantomProcessList.java),
  [developer options](https://developer.android.com/studio/debug/dev-options), [adb](https://developer.android.com/tools/adb),
  [App Standby Buckets](https://developer.android.com/topic/performance/appstandby).
- Samsung support: [Performance profile](https://www.samsung.com/us/support/answer/ANS10001953),
  [Light profile](https://www.samsung.com/ca/support/mobile-devices/light-performance-profile-on-your-samsung-galaxy),
  [background usage limits](https://www.samsung.com/us/support/answer/ANS00088422/),
  [battery protection](https://www.samsung.com/uk/support/mobile-devices/galaxy-battery-protection-feature-in-one-ui-6-1),
  [auto optimisation](https://www.samsung.com/in/support/mobile-devices/how-to-automatically-optimise-your-galaxy-device/),
  [heat](https://www.samsung.com/uk/support/mobile-devices/what-to-do-when-your-samsung-galaxy-device-heats-up-while-in-use/),
  [Pause USB PD](https://www.samsung.com/uk/support/mobile-devices/what-is-the-pause-usb-power-delivery-feature).
- ONNX Runtime: [threading doc](https://onnxruntime.ai/docs/performance/tune-performance/threading.html),
  [quantization doc](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), source
  ([threadpool.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/common/threadpool.cc),
  [cpuid_info.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/common/cpuid_info.cc),
  [convolve.cpp](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/convolve.cpp),
  [platform.cpp](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/platform.cpp)).
- Speed anchors: [Ultralytics Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/), [Zenodo 20671391](https://zenodo.org/records/20671391),
  [arXiv 2609.16085](https://arxiv.org/pdf/2609.16085v1), [ncnn benchmark](https://github.com/Tencent/ncnn/blob/master/benchmark/README.md),
  [MLPerf Mobile](https://ar5iv.labs.arxiv.org/html/2012.02328).

Secondary, used only where marked: MakeUseOf, Guiding Tech, 9to5Google, XDA, GSMArena, SamMobile, the GrapheneOS forum.
