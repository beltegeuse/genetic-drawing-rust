# Genetic Drawing

Rust version of [Anastasia Opara](https://www.anastasiaopara.com/)'s [original code](https://github.com/anopara/genetic-drawing). This project have a library form so it can be used with different interfaces. For now, you play with the code with its command line utility:

```shell
$ cd examples/cli
$ cargo run --release -- -h
Genetic Drawing 0.1
Optimize drawings

USAGE:
    cli [FLAGS] [OPTIONS] -b <brush>... -i <input> -o <output>

FLAGS:
    -c               generate color image
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -b <brush>...           path to brush image (can be specified multiple times) [required]
    -d <dist>               position distribution (uniform, gradient[:time_offset], image:path) [default: uniform]
    -g <generation>         number of generations (balance computation speed and stroke quality) [default: 30]
    -i <input>              input image to optimize [required]
    -t <iter>               number of iterations [default: 100]
    -l <last>               previous generated image (usefull for multipass drawing) [optional]
    -o <output>             output final image [required]
    -0 <scale_begin>        the brush scale at the first iteration [default: 0.3:0.7]
    -1 <scale_end>          the brush scale at the last iteration [default: 0.1:0.3]
    -s <strokes>            number of stroke that we consider each iteration [default: 10]
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
- [rayon](https://github.com/rayon-rs/rayon): Multi-thread support

## Other examples

| <center>Original</center> |  <center>Result</center> |
| ------------ | ----------- |
| ![Japan](https://raw.githubusercontent.com/beltegeuse/genetic-drawing-rust/master/assets/japan.jpg) | ![Japan Draw](https://raw.githubusercontent.com/beltegeuse/genetic-drawing-rust/master/assets/j5.png) |

## TODO

- Add GPU implementation (webgpu?)
- WASM client - Deploy on gh-pages.
- Make more friendly panic message
- Add documentation for the cli interface
