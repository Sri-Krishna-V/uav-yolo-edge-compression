# Running the Pi 5 sessions

This is a guide for running the Pi 5 from your own laptop, starting from a blank SD card. It has three parts:

1. **Set up the Pi** from a blank SD card. About 30 minutes, and done once.
2. **The fan pilot** (#29). One full cell on all 1,610 test-dev images, about 30–40 minutes. It shows whether the fan
   keeps the Pi from throttling under a long load.
3. **The speed test** (#25). Five short cells, about 15–35 minutes. It answers whether INT8 is slower than FP32 on the Pi 5.

Plan on about 2 hours, mostly waiting. Two wizards do the work from Git Bash on your laptop. Each one says what to do
at each step, and you can run it again at any time: it finds what is already done. The Pi needs no screen or keyboard.

**The one rule:** if a step's result looks wrong, stop and message Krishna before going on. The table at the end lists
what to do for the usual problems.

---

## What you need

- **The Pi 5**, its **official 27 W USB-C supply**, and the **fan**, with its 4-pin plug in the socket printed **FAN**
  on the board.
- **A microSD card of 32 GB or more**, and a card reader for your laptop. Everything on the card is erased.
- **Your Windows laptop** with Git Bash, Python and this repo, as for the Kaggle runs (`study/kaggle/README.md`, part 1).
  If your laptop is a Mac, tell Krishna first: the hotspot steps differ.
- **`pi5-files.tar`** (about 450 MB), from Krishna. It holds the model exports and the 1,610 test images. Both are too
  big for git.
- **The room's temperature.** The wizard asks for it at each session. A thermometer is best; otherwise note what you
  used, for example the AC's display.

---

## 1. Get the files (5 min)

In Git Bash:

```bash
cd uav-yolo-edge-compression        # your clone of the repo
git pull
tar -xf ~/Downloads/pi5-files.tar   # use the path where you saved it; this unpacks into study/
cd study
python -m venv .venv                # a plain Python for the wizards, with no packages
```

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

- **Stage 3, the network:** choose **1, your laptop's Mobile Hotspot**. The Pi cannot log in to the campus Wi-Fi. Turn
  the hotspot's **Power saving** off, and keep the hotspot on whenever you work with the Pi.
- **Stage 4:** press Enter for the defaults: host name `pi5`, user `pi`.
- **Stage 5, in Imager:**
  - **OS:** **Raspberry Pi OS (other) → Raspberry Pi OS (Legacy, 64-bit) Lite.** Its description says "Debian Bookworm"
    and "no desktop". Do not pick the first entry, "Raspberry Pi OS (64-bit)": the study's packages are pinned for
    Bookworm. Do not pick a 32-bit one either.
  - **User:** give it a password you will remember.
  - **Wi-Fi:** your hotspot's name and password.
  - **Remote access:** turn SSH on with public-key authentication. The wizard put your key on the clipboard, so paste it
    with Ctrl+V.
- **Stage 6:** plug the fan into the FAN socket **before** you power the Pi on. The firmware starts only a fan that is
  there at boot.
- **Stage 8 checks the Pi.** Every line must have a ✓, and a fan speed of 0 rpm is normal while the Pi is cool. A ⚠
  means stop: see the table at the end.
- **Stage 9** offers to start the pilot. Answer **y**, which is part 3 below.

---

## 3. The fan pilot (about 30–40 min)

```bash
bash pi/wizard.sh plans/pi5-fullcell.json
```

- **Stage 1** says "Active Cooler". Your fan on the FAN socket is what it checks for, so go on.
- **Stage 2:** press Enter to keep `pi5.local` and `pi`.
- **Stage 4** copies the exports and the images to the Pi. It takes a few minutes the first time.
- **Stage 5** installs the packages on the Pi the first time (about 10 minutes), then runs the setup checks. When it
  offers to run the fixes (the CPU governor and the automatic updates), answer **y**.
- **Stage 6** asks for the room's temperature, then starts the session on the Pi. It shows the progress every 30 s.
  - **If you leave the wizard open,** it goes on to stage 7 by itself when the session ends.
  - **If you want to close it,** type `l` and Enter. The session keeps running on the Pi. Run the same command later to
    come back.
  - **Do not unplug the Pi** while it runs.
- **Stage 7** copies the results to your laptop.
  - Then it tries to score them, and **"scoring failed" is expected on your laptop**, because the scorer needs Krishna's
    setup. If it asks about RAM instead, press **s**.
- **Check the summary table at the end:**
  - every row should show `ok` under `outcome`, **`none` under `throttle`**, and `ok` under `drift`;
  - **`throttled` on any row:** stop here. Do not run the speed test, and message Krishna;
  - **`re-run` under `drift`:** do exactly what the wizard prints. That is: copy the plan to a new name ending in `-2`,
    then run the wizard with it.

---

## 4. The speed test (about 15–35 min)

```bash
bash pi/wizard.sh plans/pi5-speedtest.json
```

It runs the same way as part 3, and stage 5 is quick now because the install is done. Check its summary in the same way.

---

## 5. Send the results back

From `study/`:

```bash
tar -cf ~/Desktop/pi5-results.tar bench/pi5 verification/pi5_facts.txt
```

Send `pi5-results.tar` to Krishna, by Google Drive or WhatsApp. **Do not commit these files.** The raw detections are
kept out of git, and Krishna needs them to score the runs.

With the file, tell him:

- the room temperature you entered, and how you measured it;
- anything odd, with a screenshot.

A photo of the Pi with its fan is also useful for the paper.

When you are done, shut the Pi down before you unplug it: `ssh pi@pi5.local sudo poweroff`, then wait 10 seconds.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| Imager's list has no **Raspberry Pi OS (Legacy, 64-bit) Lite** | Stop and message Krishna. Do not write a different OS. |
| Provision stage 7: no SSH to `pi5.local` | The wizard opens the hotspot settings. Find `pi5` under the connected devices, then type its IP address (`192.168.137.x`). If `pi5` is not listed after 5 minutes, the Wi-Fi name or password on the card is wrong: run `bash pi/provision.sh` again and write the card again. |
| ⚠ **No fan found** | On the Pi, run `ssh pi@pi5.local sudo poweroff`. Unplug the Pi, push the fan's plug firmly into FAN, and power the Pi on. Then run `bash pi/provision.sh` and answer **y** in stage 1, which skips to the connection. |
| ⚠ **get_throttled** is not `0x0`, or under-voltage | Use the official 27 W supply, not a phone charger. Reboot with `ssh pi@pi5.local sudo reboot`. |
| ⚠ The Pi **cannot reach PyPI** | Your laptop must have internet for the hotspot to share it. Turn the hotspot off and on, then run the stage again. |
| "**the plan is not valid**" | `study/.venv` is missing. In `study/`, run `python -m venv .venv`. |
| "**missing on this laptop: study/exports/...**" | The tar was unpacked in the wrong folder. Run `tar -xf` again from the repo's top folder, the one that contains `study/`. |
| "**lost the connection to the Pi**" | The session keeps running on the Pi. Check the hotspot, then run the same wizard command again. |
| "**scoring failed**" | Expected on your laptop. Krishna scores the runs. |
| `throttle` shows **throttled** | Stop and message Krishna. |
