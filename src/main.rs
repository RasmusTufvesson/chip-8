use std::{fs, time::{Duration, Instant}};
use sdl2::{event::Event, pixels::Color, rect::Rect};

const SCALE: u32 = 10;
const WIDTH: u8 = 64;
const HEIGHT: u8 = 32;

const FONT: [u8; 80] = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
    0x20, 0x60, 0x20, 0x20, 0x70, // 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
    0x90, 0x90, 0xF0, 0x10, 0x10, // 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
    0xF0, 0x10, 0x20, 0x40, 0x40, // 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
    0xF0, 0x90, 0xF0, 0x90, 0x90, // A
    0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
    0xF0, 0x80, 0x80, 0x80, 0xF0, // C
    0xE0, 0x90, 0x90, 0x90, 0xE0, // D
    0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
    0xF0, 0x80, 0xF0, 0x80, 0x80, // F
];

const BLACK: Color = Color::RGB(0, 0, 0);
const WHITE: Color = Color::RGB(255, 255, 255);

fn main() {
    let program = fs::read(r"programs\IBM Logo.ch8").unwrap();

    let mut interpreter = Interpreter {
        ram: RAM::new(program.as_slice()),
        display: Display::new(),
        pc: 0x200,
        ir: 0,
        stack: Stack::new(),
        delay: Timer::new(),
        sound: Timer::new(),
        variable: [0; 16],
    };

    for _ in 0..50 {
        interpreter.step();
    }

    // println!("{:?}", interpreter.display.raw);

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem.window("CHIP-8", WIDTH as u32 * SCALE, HEIGHT as u32 * SCALE)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();

    canvas.set_draw_color(BLACK);
    canvas.clear();
    canvas.present();
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut last_frame = Instant::now();
    'running: loop {
        canvas.set_draw_color(BLACK);
        canvas.clear();
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} => {
                    break 'running
                },
                _ => {}
            }
        }
        
        canvas.set_draw_color(WHITE);
        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                if interpreter.display.get(x, y) {
                    canvas.fill_rect(Rect::new(x as i32 * SCALE as i32, y as i32 * SCALE as i32, SCALE, SCALE)).unwrap();
                }
            }
        }

        canvas.present();
        let delta = Instant::now() - last_frame;
        if delta.as_secs_f64() < 1. / 60. {
            ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60) - delta);
        }
        last_frame = Instant::now()
    }
}

struct RAM {
    raw: [u8; 4096],
}

impl RAM {
    fn new(program: &[u8]) -> Self {
        let mut raw = [0; 4096];
        raw[..FONT.len()].copy_from_slice(&FONT);
        raw[0x200..0x200 + program.len()].copy_from_slice(program);
        Self {
            raw,
        }
    }

    fn fetch(&self, address: u16) -> u8 {
        self.raw[address as usize]
    }

    fn fetch_u16(&self, address: u16) -> u16 {
        (self.fetch(address) as u16) << 8 | self.fetch(address + 1) as u16
    }
}

const DISPLAY_LENGTH: usize = WIDTH as usize * HEIGHT as usize;

struct Display {
    raw: [bool; WIDTH as usize * HEIGHT as usize],
}

impl Display {
    fn new() -> Self {
        Self {
            raw: [false; DISPLAY_LENGTH],
        }
    }

    fn set(&mut self, x: u8, y: u8, value: bool, vf: &mut u8) {
        let address = y as usize * WIDTH as usize + x as usize;
        self.raw[address] ^= value;
        if !self.raw[address] {
            *vf = 1;
        }
    }

    fn get(&self, x: u8, y: u8) -> bool {
        self.raw[y as usize * WIDTH as usize + x as usize]
    }
}

struct Stack {
    raw: Vec<u16>,
}

impl Stack {
    fn new() -> Self {
        Self {
            raw: vec![],
        }
    }

    fn push(&mut self, value: u16) {
        self.raw.push(value);
    }

    fn pop(&mut self) -> u16 {
        self.raw.pop().unwrap()
    }
}

struct Timer {
    raw: u8,
}

impl Timer {
    fn new() -> Self {
        Self { raw: 0 }
    }
}

struct Interpreter {
    ram: RAM,
    display: Display,
    pc: u16,
    ir: u16,
    stack: Stack,
    delay: Timer,
    sound: Timer,
    variable: [u8; 16],
}

impl Interpreter {
    fn step(&mut self) {
        let opcode = self.ram.fetch_u16(self.pc);
        self.pc += 2;
        self.execute(Instruction {
            opcode,
            kind: (opcode & 0b1111000000000000) >> 12,
            x: ((opcode & 0b0000111100000000) >> 8) as usize,
            y: ((opcode & 0b0000000011110000) >> 4) as usize,
            n: (opcode & 0b0000000000001111) as u8,
            nn: (opcode & 0b0000000011111111) as u8,
            nnn: opcode & 0b0000111111111111,
        });
    }

    fn execute(&mut self, instruction: Instruction) {
        // println!("{:#6x} {:#b} {:#18b}", instruction.opcode, instruction.kind, instruction.opcode & 0b1111000000000000);
        match instruction.kind {
            0 => {
                match instruction.nnn {
                    0x0e0 => self.display.raw.fill(false),
                    0x0ee => self.pc = self.stack.pop(),
                    _ => {
                        println!("Machine language instruction '{:#x}'", instruction.opcode);
                    }
                }
            }
            1 => {
                self.pc = instruction.nnn;
            }
            2 => {
                self.stack.push(self.pc);
                self.pc = instruction.nnn;
            }
            3 => {
                if self.variable[instruction.x] == instruction.nn {
                    self.pc += 2;
                }
            }
            4 => {
                if self.variable[instruction.x] != instruction.nn {
                    self.pc += 2;
                }
            }
            5 => {
                if self.variable[instruction.x] == self.variable[instruction.y] {
                    self.pc += 2;
                }
            }
            9 => {
                if self.variable[instruction.x] != self.variable[instruction.y] {
                    self.pc += 2;
                }
            }
            6 => self.variable[instruction.x] = instruction.nn,
            7 => self.variable[instruction.x] += instruction.nn,
            0xA => self.ir = instruction.nnn,
            0xD => {
                let x = self.variable[instruction.x] % WIDTH;
                let y = self.variable[instruction.y] % HEIGHT;
                self.variable[0xF] = 0;
                for i in 0..instruction.n {
                    if y + i >= HEIGHT {
                        break;
                    }
                    let data = self.ram.fetch(self.ir + i as u16);
                    for j in 0..8 {
                        if x + j >= WIDTH {
                            break;
                        }
                        self.display.set(x + j, y + i, data & (0b10000000 >> j) != 0, &mut self.variable[0xF]);
                    }
                }
            }
            _ => println!("Unknown instruction kind '{:#x}'", instruction.opcode),
        }
    }
}

struct Instruction {
    opcode: u16,
    kind: u16,
    x: usize,
    y: usize,
    n: u8,
    nn: u8,
    nnn: u16,
}