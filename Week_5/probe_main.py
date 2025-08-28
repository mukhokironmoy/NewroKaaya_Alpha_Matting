# _probe_main.py
import sys, os, faulthandler
faulthandler.enable()
print("TOP: probe import ok", flush=True)
print("CWD:", os.getcwd(), "ARGV:", sys.argv, flush=True)

def main():
    print("MAIN: reached", flush=True)

if __name__ == "__main__":
    print("__name__ is:", repr(__name__), flush=True)
    main()
    print("MAIN: done", flush=True)
