# Genetic Drawing

Rust version of [Anastasia Opara](https://www.anastasiaopara.com/)'s [original code](https://github.com/anopara/genetic-drawing). This project have a library form so it can be used with different interfaces. For now, you play with the code with its command line utility:

```shell
$ cd examples/cli
$ cargo run --release -- -h
Genetic Drawing 0.1
Optimize drawings

USAGE:
    cli [OPTIONS] -b <brush>... -i <input> -o <output>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -b <brush>...           
    -d <dist>                [default: uniform]
    -g <generation>          [default: 30]
    -i <input>              Image to optimize
    -t <iter>                [default: 100]
    -l <last>               
    -o <output>             Path output final image
    -0 <scale_begin>         [default: 0.3:0.7]
    -1 <scale_end>           [default: 0.1:0.3]
    -s <strokes>             [default: 10]
```

## Example

| <center>Target image</center> | <center>Custom mask</center>|
| ------------ | ----------- |
| ![Target](https://raw.githubusercontent.com/beltegeuse/genetic-drawing-rust/master/assets/example.jpg) | ![Mask](https://raw.githubusercontent.com/beltegeuse/genetic-drawing-rust/master/assets/mask.jpg) |


With the command line interface:
1) Generate uniform strokes to fill the image
```
cargo run --release -- -i ../../assets/example.jpg -b ../../brushes/watercolor/1.jpg -b ../../brushes/watercolor/2.jpg -b ../../brushes/watercolor/3.jpg -b ../../brushes/watercolor/4.jpg -t 20 -o ../../assets/out1.png -0 0.3:0.7 -1 0.3:0.7
```
2) Make progressive strokes around images borders (where `-l` is used to load previous generated image)
```
cargo run --release -- -i ../../assets/example.jpg -b ../../brushes/watercolor/1.jpg -b ../../brushes/watercolor/2.jpg -b ../../brushes/watercolor/3.jpg -b ../../brushes/watercolor/4.jpg -t 80 -l ../../assets/out1.png -o ../../assets/out2.png -d gradient -0 0.3:0.7 -1 0.1:0.3 
```
3) Use custom mask to only refine strokes on main character (with small strokes)
```
cargo run --release -- -i ../../assets/example.jpg -b ../../brushes/watercolor/1.jpg -b ../../brushes/watercolor/2.jpg -b ../../brushes/watercolor/3.jpg -b ../../brushes/watercolor/4.jpg -t 40 -l ../../assets/out2.png -o ../../assets/out3.png -d image:../../assets/mask.jpg -0 0.1:0.3 -1 0.05:0.1 -g 50
```

| <center>Step 1</center> |  <center>Step 2</center> | <center>Step 3</center> | 
| ------------ | ----------- | ----------- |
| ![Target](https://raw.githubusercontent.com/beltegeuse/genetic-drawing-rust/master/assets/out1.png) | ![Target](https://raw.githubusercontent.com/beltegeuse/genetic-drawing-rust/master/assets/out2.png) | ![Target](https://raw.githubusercontent.com/beltegeuse/genetic-drawing-rust/master/assets/out3.png) |

## Dependencies

All the dependencies will be installed by cargo. You need to have rust installed on your computer (the simplest way is via [rustup](https://rustup.rs/)). The main dependencies are:
- [image](https://crates.io/crates/image): load and write images
- [imageproc](https://crates.io/crates/imageproc): crate that extends image processing capability (scaling, rotation, gaussian filtering)
- [rand](https://crates.io/crates/rand): random number generator

## TODO

- Make more user-friendly error/instruction for command line tool
- More control on `gradient` distribution
- Properly documenting the code
- Support color images
- Profiling and optimize code
- WASM client


