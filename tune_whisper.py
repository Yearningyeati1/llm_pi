import subprocess
import time
import itertools

WHISPER_BIN = "./build/bin/whisper-stream"
MODEL = "./models/ggml-tiny.en.bin"
WHISPER_DIR = "../whisper.cpp"


# parameter grid
PARAM_GRID = {
    "step": [2000, 3000, 4000],
    "length": [5000, 7000, 9000],
    "threads": [2, 3],
    "ac": [256, 512, 768]
}


def run_whisper(params):
    cmd = [
        "taskset", "-c", "0,1",
        WHISPER_BIN,
        "-m", MODEL,
        "--step", str(params["step"]),
        "--length", str(params["length"]),
        "-t", str(params["threads"]),
        "-ac", str(params["ac"]),
        "-c", "0"
    ]

    print("\n===============================")
    print("Testing configuration:")
    print(params)
    print("===============================\n")

    proc = subprocess.Popen(
        cmd,
        cwd=WHISPER_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    start = time.time()

    try:
        for line in proc.stdout:
            line = line.strip()

            if "]" in line:
                line = line.split("]", 1)[1].strip()

            print(line)

            if time.time() - start > 15:
                break

    finally:
        proc.terminate()

def parameter_sweep():

    keys = PARAM_GRID.keys()

    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(keys, values))

        run_whisper(params)

        print("\nPress ENTER for next test...")
        input()

if __name__ == "__main__":
    parameter_sweep()