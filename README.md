# Chip-8
This is a small chip-8 interpreter/emulator that is written in rust! The window and input handling is done with SDL2. It was written by following [this guide](https://tobiasvl.github.io/blog/write-a-chip-8-emulator/).

## Issues
* Sound does not work
* Keys are probably in the wrong order
* Some other general weirdness

## Debugging
Press escape anytime to enter debug mode, in which instructions will not run automatically, though timers will still count down. While in this mode you can press space to run a single instruction (prints information about it to the console) or 0 which prints the current state of the variable registers.