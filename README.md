# Genetic Drawing

Rust version of [Anastasia Opara](https://www.anastasiaopara.com/) [original code](https://github.com/anopara/genetic-drawing). This project can be directly usable with command line:

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

## TODO

- Make more user-friendly error/instruction for command line tool
- More control on `gradient` distribution
- Properly documenting the code
- Support color images
- Profiling and optimize code
- WASM client


