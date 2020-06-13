#[macro_use]
extern crate clap;

use clap::{App, Arg};
use rand::prelude::*;

enum DistType {
    Uniform,
    Image {
        distribution: genetic_drawing::ImageDistribution,
    },
    Gradient {
        original_image: image::DynamicImage,
        cached_distribution: Option<genetic_drawing::ImageDistribution>,
        time_offset: Option<f32>,
    },
}
impl DistType {
    fn get(&self) -> Option<&genetic_drawing::ImageDistribution> {
        match self {
            DistType::Uniform => None,
            DistType::Image { distribution } => Some(distribution),
            DistType::Gradient {
                cached_distribution,
                ..
            } => cached_distribution.as_ref(),
        }
    }
    fn update(&mut self, iter: usize, nb_iter: usize) {
        match self {
            DistType::Gradient {
                original_image,
                cached_distribution,
                time_offset,
            } => {
                let scale = match time_offset {
                    None => iter as f32 / nb_iter as f32,
                    Some(ref v) => (v + (1.0 - v) * iter as f32) / nb_iter as f32,
                };
                let t = (1.0 - scale).powi(2) * 0.25 + 0.005;
                (*cached_distribution) = Some(genetic_drawing::ImageDistribution::from_gradients(
                    original_image,
                    t,
                ));
            }
            _ => {}
        }
    }
}

fn parse_range(v: String) -> genetic_drawing::ScaleRange {
    match v.split(":").into_iter().map(|v| v).collect::<Vec<_>>()[..] {
        [v0, v1] => {
            let v0 = v0
                .parse::<f32>()
                .expect(&format!("Impossible to parse range value: {}", v0));
            let v1 = v1
                .parse::<f32>()
                .expect(&format!("Impossible to parse range value: {}", v1));
            if v0 >= 0.0 && v1 >= 0.0 && v0 > 1.0 && v1 > 1.0 {
                panic!(
                    "Wrong range values [{}, {}]: needs to between 1.0 and 0.0 (excluded)",
                    v0, v1
                );
            }
            if v0 > v1 {
                panic!(
                    "Wrong range values [{}, {}]: min value is bigger than max value",
                    v0, v1
                );
            }
            genetic_drawing::ScaleRange { min: v0, max: v1 }
        }
        _ => panic!("Wrong parameter value for scale begin"),
    }
}

fn main() {
    let matches = App::new("Genetic Drawing")
        .version("0.1")
        .about("Optimize drawings")
        .arg(
            Arg::with_name("input")
                .help("input image to optimize [required]")
                .short("i")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .help("output final image [required]")
                .short("o")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("scale_begin")
                .help("the brush scale at the first iteration")
                .short("0")
                .takes_value(true)
                .default_value("0.3:0.7"),
        )
        .arg(
            Arg::with_name("scale_end")
                .help("the brush scale at the last iteration")
                .short("1")
                .takes_value(true)
                .default_value("0.1:0.3"),
        )
        .arg(
            Arg::with_name("iter")
                .help("number of iterations")
                .short("t")
                .takes_value(true)
                .default_value("100"),
        )
        .arg(
            Arg::with_name("strokes")
                .help("number of stroke that we consider each iteration")
                .short("s")
                .takes_value(true)
                .default_value("10"),
        )
        .arg(
            Arg::with_name("generation")
                .help("number of generations (balance computation speed and stroke quality)")
                .takes_value(true)
                .short("g")
                .default_value("30"),
        )
        .arg(
            Arg::with_name("dist")
                .help("position distribution (uniform, gradient[:time_offset], image:path)")
                .takes_value(true)
                .short("d")
                .default_value("uniform"),
        )
        .arg(
            Arg::with_name("brush")
                .help("path to brush image (can be specified multiple times) [required]")
                .short("b")
                .takes_value(true)
                .required(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("last")
                .help("previous generated image (usefull for multipass drawing) [optional]")
                .short("l")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("progressive_dir")
                .help("directory where to save each iteration (path[:offset]) [optional]")
                .short("p")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("color")
                .help("generate color image")
                .short("c"),
        )
        .get_matches();

    let now = std::time::Instant::now();

    // Create random number
    let mut rng = thread_rng();
    let with_color = matches.is_present("color");

    // Generate the main object & Register brushes
    let filename_out = value_t_or_exit!(matches.value_of("output"), String);
    let filename = value_t_or_exit!(matches.value_of("input"), String);
    let img = image::open(&filename).expect(&format!("Impossible to open {}", filename));
    let mut gen = {
        if with_color {
            genetic_drawing::GeneticDrawing::load(genetic_drawing::Image::Color(img.to_rgb()))
        } else {
            genetic_drawing::GeneticDrawing::load(genetic_drawing::Image::Gray(img.to_luma()))
        }
    };
    let brushes = values_t_or_exit!(matches.values_of("brush"), String);
    if brushes.is_empty() {
        // Note that this error message might be not necessary
        // as the mark brush argument as mandantory
        panic!("You need to specify at least one brush image (via -b)");
    }
    for b in brushes {
        let img = image::open(&b).expect(&format!("Impossible to open {}", b));
        gen.register_brush(img);
    }

    // Brushes range
    let scale_begin = parse_range(value_t_or_exit!(matches.value_of("scale_begin"), String));
    let scale_end = parse_range(value_t_or_exit!(matches.value_of("scale_end"), String));
    gen.bruch_range = (scale_begin, scale_end);

    // Type of stroke position approaches
    let dist = value_t_or_exit!(matches.value_of("dist"), String);
    let dist = dist.split(":").into_iter().map(|v| v).collect::<Vec<_>>();
    let mut dist = match &dist[..] {
        ["uniform"] => DistType::Uniform,
        ["image", v] => {
            let dist = image::open(&v)
                .expect(&format!("impossible to open {}", v))
                .to_luma();
            DistType::Image {
                distribution: genetic_drawing::ImageDistribution::from_image(&dist),
            }
        }
        ["gradient"] => {
            let original_image = gen.img_gradient.mag.to_image();
            DistType::Gradient {
                original_image,
                cached_distribution: None,
                time_offset: None,
            }
        }
        ["gradient", v] => {
            // A time offset is provided
            let time_offset = v.parse::<f32>().expect(
                "Impossible to parse gradient time (need to be a float between 0.0 and 1.0",
            );
            if time_offset < 0.0 || time_offset > 1.0 {
                panic!("Impossible to parse gradient time (need to be a float between 0.0 and 1.0");
            }

            let original_image = gen.img_gradient.mag.to_image();
            DistType::Gradient {
                original_image,
                cached_distribution: None,
                time_offset: Some(time_offset),
            }
        }
        _ => panic!("Wrong dist option: {:?}", dist),
    };

    // Output directory for each iteration
    let progressive_dir = match matches.value_of("progressive_dir") {
        None => None,
        Some(s) => {
            let s = s.split(":").into_iter().map(|v| v).collect::<Vec<_>>();
            match s[..] {
                [s] => Some((s, 0)),
                [s, i] => {
                    let i = i
                        .parse::<usize>()
                        .expect(&format!("Impossible to parse {} as usize", i));
                    Some((s, i))
                }
                _ => panic!("Impossible to parse progressive_dir argument"),
            }
        }
    };

    // DNA
    let nb_iter = value_t_or_exit!(matches.value_of("iter"), usize);
    let nb_strokes = value_t_or_exit!(matches.value_of("strokes"), usize);
    let nb_gen = value_t_or_exit!(matches.value_of("generation"), usize);

    // Run
    let mut last_image = match matches.value_of("last") {
        None => None,
        Some(v) => {
            let img = image::open(v).expect(&format!("Impossible to open {}", v));
            if with_color {
                Some(genetic_drawing::Image::Color(img.to_rgb()))
            } else {
                Some(genetic_drawing::Image::Gray(img.to_luma()))
            }
        }
    };
    let mut pb = pbr::ProgressBar::new(nb_iter as u64);
    for i in 0..nb_iter {
        dist.update(i, nb_iter);
        let mut dna = genetic_drawing::DNAContext::new(
            &gen,
            nb_strokes,
            &mut rng,
            (i as f32 + 0.5) / nb_iter as f32,
            last_image.as_ref(),
            dist.get(),
        );
        dna.iterate(nb_gen, &mut rng);
        last_image = Some(dna.to_image());

        // Save the image if necessary
        if let Some((outdir, iter)) = progressive_dir {
            let outfile = std::path::Path::new(outdir).join(&format!("{:0>5}.png", iter + i));
            last_image
                .as_ref()
                .unwrap()
                .clone()
                .as_dynamic_image()
                .save(&outfile)
                .expect(&format!("Impossible to save file {:?}", outfile));
        }
        pb.inc();
    }

    if let Some(last_image) = last_image {
        last_image
            .as_dynamic_image()
            .save(&filename_out)
            .expect(&format!("Impossible to save {}", filename_out));
    }

    let last = std::time::Instant::now();
    println!("Time for processing: {} sec", (last - now).as_secs());
}
