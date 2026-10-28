# Running the Pi 5 sessions

This is a guide for running the Pi 5 from your own laptop, starting from a blank SD card. Parts 1–5 are done; the
matrix sessions are [part 6](#6-the-matrix-sessions-37). The first parts:

1. **Set up the Pi** from a blank SD card. About 30 minutes, and done once.
2. **The fan pilot** (#29). One full cell on all 1,610 test-dev images, about 30–40 minutes. It shows whether the fan
   keeps the Pi from throttling under a long load.
3. **The speed test** (#25). Five short cells, about 15–35 minutes. It answers whether INT8 is slower than FP32 on the Pi 5.

Plan on 2–2.5 hours, mostly waiting. Two wizards do the work from Git Bash on your laptop. Each one says what to do at
each step, and you can run it again at any time: it finds what is already done and goes on from there. The Pi needs no
screen or keyboard.

Every decision you might meet is answered below, so you can finish on your own. For a problem, look in the table at the
end first.

---

## What you need

- **The Pi 5**, its **official 27 W USB-C supply**, and the **fan**, with its 4-pin plug in the socket printed **FAN**
  on the board.
- **A microSD card of 32 GB or more**, and a card reader for your laptop. Everything on the card is erased.
- **A Windows laptop** with Git Bash, Python and this repo, as for the Kaggle runs (`study/kaggle/README.md`, part 1).
  It needs a recent Git for Windows; the 2.4x or 2.5x versions work.
- **`pi5-files.tar`** (about 450 MB), from Krishna. It holds the model exports and the 1,610 test images. Both are too
  big for git.
- **A network for the Pi:** your laptop's Mobile Hotspot, or your phone's hotspot as a backup. The Pi cannot log in to
  the campus Wi-Fi.
- **The room's temperature.** The wizard asks for it at each session. A thermometer near the Pi is best; otherwise use
  the AC's display or a weather app, and remember what you used.

---

## 1. Get the files (5 min)

In Git Bash:

```bash
cd uav-yolo-edge-compression        # your clone of the repo
git pull                            # if it refuses because of local changes: git stash, then git pull again
tar -xf ~/Downloads/pi5-files.tar   # use the path where you saved it; this unpacks into study/
cd study
python -m venv .venv                # a plain Python for the wizards, with no packages
```

If `python -m venv .venv` says that Python was not found, run `py -m venv .venv` instead.

Check that the files arrived:

```bash
ls exports/yolo11s-640-s0/640                  # ncnn-fp32  onnxruntime-fp32  onnxruntime-int8  openvino-fp32  openvino-int8
ls datasets/VisDrone/images/test | wc -l       # 1610
```

Every command below runs from `study/`.

---

## 2. Set up the Pi (about 30 min, once)

```bash
bash pi/provision.sh
```

It has 9 stages. The ones that need care:

- **Stage 3, the network:**
  - **Choose 1, your laptop's Mobile Hotspot.** Turn the hotspot's **Power saving** off, and keep the hotspot on
    whenever you work with the Pi.
  - **If Windows will not turn the hotspot on,** use your phone's hotspot instead. Connect the laptop to the phone's
    hotspot too, then choose **2**.
- **Stage 4:** press Enter for the defaults: host name `pi5`, user `pi`.
- **Stage 5, in Imager:**
  - **OS:** **Raspberry Pi OS (other) → Raspberry Pi OS (Legacy, 64-bit) Lite.** Its description says "Debian Bookworm"
    and "no desktop". Do not pick the first entry, "Raspberry Pi OS (64-bit)": the study's packages are pinned for
    Bookworm. Do not pick a 32-bit one either.
  - **User:** give it a password you will remember.
  - **Wi-Fi:** the name and password of the hotspot you chose in stage 3.
  - **Remote access:** turn SSH on with public-key authentication. The wizard put your key on the clipboard, so paste it
    with Ctrl+V.
- **Stage 6:** plug the fan into the FAN socket **before** you power the Pi on. The firmware starts only a fan that is
  there at boot.
- **Stage 7** finds the Pi by its name. If that fails, it searches the network for a Raspberry Pi by itself.
- **Stage 8 checks the Pi.** Every line must have a ✓, and a fan speed of 0 rpm is normal while the Pi is cool. Each ⚠
  has its fix in the table at the end.
- **Stage 9** offers to start the pilot. Answer **y**, which is part 3 below.

---

## 3. The fan pilot (about 30–40 min)

```bash
bash pi/wizard.sh plans/pi5-fullcell.json
```

- **Stage 1** says "Active Cooler". Your fan on the FAN socket is what it checks for, so go on.
- **Stage 2:** press Enter to keep the host and the user.
- **Stage 4** copies the exports and the images to the Pi. It takes a few minutes the first time.
- **Stage 5** installs the packages on the Pi the first time (about 10 minutes), then runs the setup checks. When it
  offers to run the fixes (the CPU governor and the automatic updates), answer **y**.
- **Stage 6** asks for the room's temperature, then starts the session on the Pi. It shows the progress every 30 s.
  - **If you leave the wizard open,** it goes on to stage 7 by itself when the session ends.
  - **If you want to close it,** type `l` and Enter. The session keeps running on the Pi. Run the same command later to
    come back.
  - **Do not unplug the Pi** while it runs. Your laptop may sleep.
- **Stage 7** copies the results to your laptop. Then it tries to score them, and **"scoring failed" is expected**:
  scoring needs Krishna's setup, and he scores the runs. If it asks about RAM instead, press **s**. The "score later"
  line it then lists under "still to do by hand" is for Krishna too.

**Did the pilot pass?** Run this:

```bash
grep -q 'complete throttled' bench/pi5/pi5-fullcell/summary.csv && echo "THROTTLED: do 3b" || echo "PASS: go to part 4"
```

**If the summary's `drift` column says `re-run`:** do what the wizard prints, once. That is, copy the plan to a new name
ending in `-2` and run the wizard with it. If the `-2` session says `re-run` too, go on to part 4 anyway, and mention it
when you send the results.

### 3b. Only if the pilot throttled: lock the CPU clock and run it again

The study's rule for this case (#29, #8) is to lock the Pi's clock at the highest of 2000, 1800 and 1500 MHz that does
not throttle. Then every Pi 5 run keeps that lock. First, try 2000:

```bash
. pi/.env    # loads the Pi's host and user
ssh "$PI_USER@$PI_HOST" "sudo sed -i '/^arm_freq=/d' /boot/firmware/config.txt; printf '[all]\narm_freq=2000\n' | sudo tee -a /boot/firmware/config.txt; sudo reboot"
```

The connection closes as the Pi reboots, which is normal. Wait one minute, then run the pilot again under a new name:

```bash
cp plans/pi5-fullcell.json plans/pi5-fullcell-2000.json
bash pi/wizard.sh plans/pi5-fullcell-2000.json   # stage 5 offers the governor fix again after the reboot: answer y
grep -q 'complete throttled' bench/pi5/pi5-fullcell-2000/summary.csv && echo "THROTTLED" || echo "PASS"
```

- **PASS:** keep the lock, and go to part 4.
- **THROTTLED:** do the same with `1800` in all four places: `arm_freq=1800`, `pi5-fullcell-1800.json`, and so on.
  Then, if needed, with `1500`.
- **Throttled even at 1500:** the fan is not doing its job. Check that it spins while a session runs, then do part 4
  anyway.

Whatever you get, say in your message which lock you ended with.

---

## 4. The speed test (about 15–35 min)

```bash
bash pi/wizard.sh plans/pi5-speedtest.json
```

It runs the same way as part 3, with the clock lock if part 3b set one. Stage 5 is quick now, because the install is
done. If a cell throttles here, the session runs that cell again by itself after the Pi cools down.

---

## 5. Send the results back

From `study/`:

```bash
tar -cf ~/pi5-results.tar bench/pi5 verification/pi5_facts.txt plans/pi5-*.json
explorer.exe "$(cygpath -w ~)"    # opens the folder that holds pi5-results.tar
```

Send `pi5-results.tar` to Krishna, by Google Drive or WhatsApp. **Do not commit these files.** The raw detections are
kept out of git, and Krishna needs them to score the runs.

With the file, tell him:

- the room temperature you entered, and how you measured it;
- the clock lock from part 3b, or "no lock";
- anything odd, with a screenshot.

A photo of the Pi with its fan is also useful for the paper.

When you are done, shut the Pi down before you unplug it:

```bash
. pi/.env; ssh "$PI_USER@$PI_HOST" sudo poweroff
```

Then wait 10 seconds before unplugging.

---

## 6. The matrix sessions (#37)

Parts 1–5 are done. This part runs the study's 63 cells in 6 sessions, about 20 hours of Pi time in all. The Pi keeps
`arm_freq=2000`: do not change `config.txt` again. Stage 3 of the wizard stops if the lock is missing.

**Before each batch of sessions,** Krishna sends a tar of exports by Google Drive. From the repo's top folder:

```bash
git pull                                       # the plans and the wizard
tar -xf ~/Downloads/pi5-matrix-exports-1.tar   # use its real name; it unpacks into study/exports/
cd study
```

**Run the sessions in this order,** one at a time. The wizard copies each session's exports to the Pi in stage 4.

| Order | Command | Pi time |
|---|---|---|
| 1 | `bash pi/wizard.sh plans/pi5-matrix-1-yolo11s.json` | about 4.8 h |
| 2 | `bash pi/wizard.sh plans/pi5-matrix-2-yolo26s.json` | about 3.3 h |
| 3 | `bash pi/wizard.sh plans/pi5-matrix-3-yolo26s-compressed.json` (needs your 4 checkpoints, in a later tar) | about 2.2 h |
| 4 | `bash pi/wizard.sh plans/pi5-matrix-4-1280-baselines.json` | about 3.5 h |
| 5 | `bash pi/wizard.sh plans/pi5-matrix-5-800.json` | about 2.1 h |
| 6 | `bash pi/wizard.sh plans/pi5-matrix-6-1120.json` | about 3.7 h |

- A session needs you only to start it and to bring its results home. Type `l` in stage 6 to leave it running, and run
  the same command again when it is done. Start the next session right after.
- If a session's exports are not on your laptop yet ("missing on this laptop"), its tar has not come yet: run the next
  one whose exports you have.
- If time runs out, the sessions at the end of the table are the ones dropped. Do not skip ahead.
- "scoring failed" in stage 7 is still expected.

**After each session,** from `study/`, push the small files as last time and send the detections, which git leaves out:

```bash
git add bench/pi5/pi5-matrix-1-yolo11s && git commit -m "Pi 5 matrix session 1" && git push
tar -czf ~/pi5-matrix-1-yolo11s.tgz bench/pi5/pi5-matrix-1-yolo11s   # about 250 MB; upload it to Google Drive
```

Change the session's name in both lines. If a session says `re-run` in its drift column, tell Krishna before starting
the next one.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| Windows will not turn the Mobile hotspot on | Use your phone's hotspot. Connect the laptop to it too. Then run `bash pi/provision.sh` again: write the card with the phone hotspot's name and password, and choose **2** in stage 3. |
| Provision stage 7 cannot find the Pi | It searches the network by itself. Press Enter at its prompt to search again after a minute. If nothing turns up after 5 minutes, the Wi-Fi name or password on the card is wrong: run `bash pi/provision.sh` again and write the card again. |
| ⚠ **No fan found** | Run `. pi/.env; ssh "$PI_USER@$PI_HOST" sudo poweroff`. Unplug the Pi, push the fan's plug firmly into FAN, and power the Pi on. Then run `bash pi/provision.sh` and answer **y** in stage 1, which skips to the connection. |
| ⚠ **get_throttled** is not `0x0`, or under-voltage | Use the official 27 W supply, not a phone charger. Then reboot with `. pi/.env; ssh "$PI_USER@$PI_HOST" sudo reboot`, and run `bash pi/provision.sh` again with **y** in stage 1. |
| ⚠ The Pi **cannot reach PyPI** | The hotspot shares your laptop's internet, so your laptop must be online. Turn the hotspot off and on, then run `bash pi/provision.sh` again with **y** in stage 1. |
| ⚠ **SD card** smaller than 28 GB | It is fine for these runs. Mention it when you send the results. |
| "**the plan is not valid**" | `study/.venv` is missing. In `study/`, run `python -m venv .venv`, or `py -m venv .venv`. |
| "**missing on this laptop: study/exports/...**" | The tar was unpacked in the wrong folder. Run `tar -xf` again from the repo's top folder, the one that contains `study/`. |
| "**lost the connection to the Pi**" | The session keeps running on the Pi. Check the hotspot, then run the same wizard command again. |
| "**the install failed**" (stage 5) | It is usually a dropped connection. Check that the Pi has internet, as in the PyPI row above, then run the same command again. |
| "**scoring failed**" | Expected on your laptop. Krishna scores the runs. |
| "**the session stopped as failed**" | Run the same command again: it resumes, and the cells that finished are kept. |
| Any other error | Run the same command once more; the wizards pick up where they stopped. If it stops the same way again, take a screenshot, do part 5 with what you have, and send both. |
