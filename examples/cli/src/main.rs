#[macro_use]
extern crate clap;

use clap::{App, Arg};
use rand::prelude::*;

enum DistType {
    Uniform,
    Image {
        distribution: genetic_drawing::Distribution,
    },
    Gradient {
        original_image: image::DynamicImage,
        cached_distribution: Option<genetic_drawing::Distribution>,
        time_offset: Option<f32>,
    },
}
impl DistType {
    fn get(&self) -> Option<&genetic_drawing::Distribution> {
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
                (*cached_distribution) = Some(genetic_drawing::Distribution::from_gradients(
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
                .help("Image to optimize")
                .short("i")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .help("Path output final image")
                .short("o")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("scale_begin")
                .short("0")
                .takes_value(true)
                .default_value("0.3:0.7"),
        )
        .arg(
            Arg::with_name("scale_end")
                .short("1")
                .takes_value(true)
                .default_value("0.1:0.3"),
        )
        .arg(
            Arg::with_name("iter")
                .short("t")
                .takes_value(true)
                .default_value("100"),
        )
        .arg(
            Arg::with_name("strokes")
                .short("s")
                .takes_value(true)
                .default_value("10"),
        )
        .arg(
            Arg::with_name("generation")
                .takes_value(true)
                .short("g")
                .default_value("30"),
        )
        .arg(
            Arg::with_name("dist")
                .takes_value(true)
                .short("d")
                .default_value("uniform"),
        )
        .arg(
            Arg::with_name("brush")
                .short("b")
                .takes_value(true)
                .required(true)
                .multiple(true),
        )
        .arg(Arg::with_name("last").short("l").takes_value(true))
        .get_matches();

    let now = std::time::Instant::now();

    // Create random number
    let mut rng = thread_rng();

    // Generate the main object & Register brushes
    let filename_out = value_t_or_exit!(matches.value_of("output"), String);
    let filename = value_t_or_exit!(matches.value_of("input"), String);
    let mut gen = genetic_drawing::GeneticDrawing::load(&filename);
    let brushes = values_t_or_exit!(matches.values_of("brush"), String);
    for b in brushes {
        gen.register_brush(&b);
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
                distribution: genetic_drawing::Distribution::from_image(&dist),
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

    // DNA
    let nb_iter = value_t_or_exit!(matches.value_of("iter"), usize);
    let nb_strokes = value_t_or_exit!(matches.value_of("strokes"), usize);
    let nb_gen = value_t_or_exit!(matches.value_of("generation"), usize);

    // Run
    let mut last_image = match matches.value_of("last") {
        None => None,
        Some(v) => Some(
            image::open(v)
                .expect(&format!("Impossible to open {}", v))
                .to_luma(),
        ),
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
        pb.inc();
    }

    if let Some(last_image) = last_image {
        last_image
            .save(&filename_out)
            .expect(&format!("Impossible to save {}", filename_out));
    }

    let last = std::time::Instant::now();
    println!("Time for processing: {} sec", (last - now).as_secs());
}
