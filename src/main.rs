use std::{fs, time::{Duration, Instant}};
use rand::{rngs::ThreadRng, thread_rng, Rng};
use sdl2::{event::Event, keyboard::Scancode, pixels::Color, rect::Rect};

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

const SCANCODES: [Scancode; 16] = [
    Scancode::Num1, Scancode::Num2, Scancode::Num3, Scancode::Num4,
    Scancode::Q, Scancode::W, Scancode::E, Scancode::R,
    Scancode::A, Scancode::S, Scancode::D, Scancode::F,
    Scancode::Z, Scancode::X, Scancode::C, Scancode::V,
    ];

const BLACK: Color = Color::RGB(0, 0, 0);
const WHITE: Color = Color::RGB(255, 255, 255);

fn main() {
    let program = fs::read(r"programs\test_opcode.ch8").unwrap();

    let mut interpreter = Interpreter {
        ram: RAM::new(program.as_slice()),
        display: Display::new(),
        pc: 0x200,
        ir: 0,
        stack: Stack::new(),
        delay: Timer::new(),
        sound: Timer::new(),
        variable: [0; 16],
        keypad: Keypad { keys: [false; 16] },
        rng: thread_rng(),

        original_shift_instruction: false,
        original_jump_with_offset_instruction: true,
        original_add_to_index_instruction: true,
        original_store_and_load_instruction: false,

        debug: false,
    };

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
    let mut delta = 0.0;
    'running: loop {
        canvas.set_draw_color(BLACK);
        canvas.clear();
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} => {
                    break 'running
                }
                Event::KeyDown { scancode, .. } => {
                    if let Some(code) = scancode {
                        if let Some((index, _)) = SCANCODES.iter().enumerate().find(|(_, x)| x == &&code) {
                            interpreter.keypad.keys[index] = true;
                        } else if code == Scancode::Escape {
                            interpreter.debug = !interpreter.debug;
                        } else if interpreter.debug {
                            match code {
                                Scancode::Space => interpreter.step(),
                                Scancode::Num0 => println!("{:?}", interpreter.variable),
                                _ => {}
                            }
                        }
                    }
                }
                Event::KeyUp { scancode, .. } => {
                    if let Some(code) = scancode {
                        if let Some((index, _)) = SCANCODES.iter().enumerate().find(|(_, x)| x == &&code) {
                            interpreter.keypad.keys[index] = false;
                        }
                    }
                }
                _ => {}
            }
        }

        if interpreter.delay.raw != 0 {
            interpreter.delay.raw -= 1;
        }
        if interpreter.sound.raw != 0 {
            interpreter.sound.raw -= 1;
        }

        if !interpreter.debug {
            for _ in 0..(delta / (1. / 700.)) as usize {
                interpreter.step();
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
        let diff = Instant::now() - last_frame;
        if diff.as_secs_f64() < 1. / 60. {
            ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60) - diff);
        }
        delta = (Instant::now() - last_frame).as_secs_f64();
        last_frame = Instant::now()
    }
}

struct RAM {
    raw: [u8; 4096],
    last_program_address: u16,
}

impl RAM {
    fn new(program: &[u8]) -> Self {
        let mut raw = [0; 4096];
        raw[..FONT.len()].copy_from_slice(&FONT);
        raw[0x200..0x200 + program.len()].copy_from_slice(program);
        Self {
            raw,
            last_program_address: 0x200 + program.len() as u16,
        }
    }

    fn fetch(&self, address: u16) -> u8 {
        self.raw[address as usize]
    }

    fn fetch_u16(&self, address: u16) -> u16 {
        (self.fetch(address) as u16) << 8 | self.fetch(address + 1) as u16
    }

    fn set(&mut self, address: u16, value: u8) {
        self.raw[address as usize] = value;
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

struct Keypad {
    keys: [bool; 16]
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
    keypad: Keypad,
    rng: ThreadRng,
    original_shift_instruction: bool,
    original_jump_with_offset_instruction: bool,
    original_add_to_index_instruction: bool,
    original_store_and_load_instruction: bool,
    debug: bool,
}

impl Interpreter {
    fn step(&mut self) {
        if self.pc <= self.ram.last_program_address {
            let opcode = self.ram.fetch_u16(self.pc);
            if self.debug {
                print!("{:#x}: {:#x}: ", self.pc, opcode);
            }
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
    }

    fn execute(&mut self, instruction: Instruction) {
        match instruction.kind {
            0 => {
                match instruction.nnn {
                    0x0e0 => {
                        self.display.raw.fill(false);
                        if self.debug { println!("clear") }
                    }
                    0x0ee => {
                        self.pc = self.stack.pop();
                        if self.debug { println!("return") }
                    }
                    _ => println!("Machine language instruction '{:#x}'", instruction.opcode),
                }
            }
            1 => {
                self.pc = instruction.nnn;
                if self.debug { println!("jump {:#x}", self.pc) }
            }
            2 => {
                self.stack.push(self.pc);
                self.pc = instruction.nnn;
                if self.debug { println!("call {:#x}", self.pc) }
            }
            3 => {
                if self.variable[instruction.x] == instruction.nn {
                    self.pc += 2;
                }
                if self.debug { println!("eq {:#x} {:#x}", instruction.x, instruction.nn) }
            }
            4 => {
                if self.variable[instruction.x] != instruction.nn {
                    self.pc += 2;
                }
                if self.debug { println!("neq {:#x} {:#x}", instruction.x, instruction.nn) }
            }
            5 => {
                if self.variable[instruction.x] == self.variable[instruction.y] {
                    self.pc += 2;
                }
                if self.debug { println!("eqv {:#x} {:#x}", instruction.x, instruction.y) }
            }
            9 => {
                if self.variable[instruction.x] != self.variable[instruction.y] {
                    self.pc += 2;
                }
                if self.debug { println!("neqv {:#x} {:#x}", instruction.x, instruction.y) }
            }
            6 => {
                self.variable[instruction.x] = instruction.nn;
                if self.debug { println!("setvn {:#x} {:#x}", instruction.x, instruction.nn) }
            }
            7 => {
                (self.variable[instruction.x], _) = self.variable[instruction.x].overflowing_add(instruction.nn);
                if self.debug { println!("addvn {:#x} {:#x}", instruction.x, instruction.nn) }
            }
            8 => {
                match instruction.n {
                    0 => self.variable[instruction.x] = self.variable[instruction.y],
                    1 => self.variable[instruction.x] |= self.variable[instruction.y],
                    2 => self.variable[instruction.x] &= self.variable[instruction.y],
                    3 => self.variable[instruction.x] ^= self.variable[instruction.y],
                    4 => {
                        let (value, overflow) = self.variable[instruction.x].overflowing_add(self.variable[instruction.y]);
                        self.variable[0xF] = if overflow { 1 } else { 0 };
                        self.variable[instruction.x] = value;
                    }
                    5 => {
                        let (value, overflow) = self.variable[instruction.x].overflowing_sub(self.variable[instruction.y]);
                        self.variable[0xF] = if overflow { 0 } else { 1 };
                        self.variable[instruction.x] = value;
                    }
                    7 => {
                        let (value, overflow) = self.variable[instruction.y].overflowing_sub(self.variable[instruction.x]);
                        self.variable[0xF] = if overflow { 0 } else { 1 };
                        self.variable[instruction.x] = value;
                    }
                    6 => {
                        if self.original_shift_instruction {
                            self.variable[instruction.x] = self.variable[instruction.y];
                        }
                        self.variable[0xF] = self.variable[instruction.x] & 0b00000001;
                        self.variable[instruction.x] >>= 1;
                    }
                    0xE => {
                        if self.original_shift_instruction {
                            self.variable[instruction.x] = self.variable[instruction.y];
                        }
                        self.variable[0xF] = (self.variable[instruction.x] & 0b10000000) >> 7;
                        self.variable[instruction.x] <<= 1;
                    }
                    _ => println!("Unknown logic/arithmetic instruction '{:#x}'", instruction.opcode),
                }
                if self.debug { println!("logic/arithmetic") }
            }
            0xA => {
                self.ir = instruction.nnn;
                if self.debug { println!("seti {:#x}", instruction.nn) }
            }
            0xB => {
                if self.original_jump_with_offset_instruction {
                    self.pc = instruction.nnn + self.variable[0] as u16;
                } else {
                    self.pc = instruction.nnn + self.variable[instruction.x] as u16;
                }
                if self.debug { println!("jumpo {:#x}", instruction.x) }
            }
            0xC => {
                self.variable[instruction.x] = self.rng.gen::<u8>() & instruction.nn;
                if self.debug { println!("rand {:#x}", instruction.x) }
            }
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
                if self.debug { println!("draw {:#x} {:#x} {:#x}", instruction.x, instruction.y, instruction.n) }
            }
            0xE => {
                match instruction.nn {
                    0x9e => {
                        if self.keypad.keys[instruction.x] {
                            self.pc += 2;
                        }
                        if self.debug { println!("prsd {:#x}", instruction.x) }
                    }
                    0xa1 => {
                        if !self.keypad.keys[instruction.x] {
                            self.pc += 2;
                        }
                        if self.debug { println!("nprsd {:#x}", instruction.x) }
                    }
                    _ => println!("Unknown input instruction '{:#x}'", instruction.opcode),
                }
            }
            0xF => {
                match instruction.nn {
                    0x07 => {
                        self.variable[instruction.x] = self.delay.raw;
                        if self.debug { println!("setvd") }
                    }
                    0x15 => {
                        self.delay.raw = self.variable[instruction.x];
                        if self.debug { println!("setdv") }
                    }
                    0x18 => {
                        self.sound.raw = self.variable[instruction.x];
                        if self.debug { println!("setsv") }
                    }
                    0x1e => {
                        self.ir += self.variable[instruction.x] as u16;
                        if !self.original_add_to_index_instruction && self.ir & 0b1000 != 0 {
                            self.variable[0xF] = 1;
                        }
                        if self.debug { println!("addiv") }
                    }
                    0x0a => {
                        let mut continue_execution = false;
                        for k in 0..16 {
                            if self.keypad.keys[k] {
                                continue_execution = true;
                                self.variable[instruction.x] = k as u8;
                            }
                        }
                        if !continue_execution {
                            self.pc -= 2;
                        }
                        if self.debug { println!("wait") }
                    }
                    0x29 => {
                        self.ir = ((self.variable[instruction.x] & 0b00001111) * 5) as u16;
                        if self.debug { println!("setidgt") }
                    }
                    0x33 => {
                        let number = self.variable[instruction.x];
                        self.ram.set(self.ir, number / 100);
                        self.ram.set(self.ir + 1, (number % 100) / 10);
                        self.ram.set(self.ir + 2, number % 10);
                        if self.debug { println!("dgt") }
                    }
                    0x55 => {
                        for register in 0..=instruction.x {
                            self.ram.set(self.ir + register as u16, self.variable[register]);
                        }
                        if self.original_store_and_load_instruction {
                            self.ir += instruction.x as u16 + 1;
                        }
                        if self.debug { println!("setmv") }
                    }
                    0x65 => {
                        for register in 0..=instruction.x {
                            self.variable[register] = self.ram.fetch(self.ir + register as u16);
                        }
                        if self.original_store_and_load_instruction {
                            self.ir += instruction.x as u16 + 1;
                        }
                        if self.debug { println!("setvm") }
                    }
                    _ => println!("Unknown instruction '{:#x}'", instruction.opcode),
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