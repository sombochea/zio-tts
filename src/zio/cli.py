"""zio speak "text" out.wav --run <dir> [--voice id] [--speed 1.0] [--expressive 0.667]  |  zio transcribe audio.wav --model <dir_or_hf_id> [--lang km|en]"""
import argparse


def main():
    ap = argparse.ArgumentParser(prog="zio")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speak")
    s.add_argument("text"), s.add_argument("out"), s.add_argument("--run", required=True), s.add_argument("--voice")
    s.add_argument("--speed", type=float, default=1.0), s.add_argument("--expressive", type=float, default=0.667)
    t = sub.add_parser("transcribe")
    t.add_argument("audio"), t.add_argument("--model", required=True), t.add_argument("--lang", default="km")
    a = ap.parse_args()
    if a.cmd == "speak":
        from zio.tts.infer import synthesize
        synthesize(a.run, a.text, a.out, a.voice, a.speed, a.expressive)
    else:
        from zio.stt.infer import transcribe
        print(transcribe(a.model, a.audio, a.lang))


if __name__ == "__main__":
    main()
