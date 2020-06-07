extern crate genetic_drawing_rust;
use rand::prelude::*;

fn main() {
    // Statements here are executed when the compiled binary is called
    let mut rng = thread_rng();

    // Print text to the console
    let mut gen = genetic_drawing_rust::GeneticDrawing::load("./data/example.jpg");
    let (img_mag, img_ang) = gen.img_gradient.to_image();
    img_ang.save("ang.png").unwrap();
    img_mag.save("mag.png").unwrap();
    {
        genetic_drawing_rust::Distribution::from_gradients(&img_mag, 0.005).to_image().save("test.png").unwrap();
    }

    // Register strokes
    println!("Brushes...");
    for i in 0..4 {
        gen.register_brush(&format!("./brushes/watercolor/{}.jpg", i + 1));
    }

    // Brushes dumps
    for i in 0..gen.brushes.len() {
        gen.brushes[i].save(&format!("b_{}.png", i)).unwrap();
    }

    println!("DNA...");
    let mut images = vec![];
    const NB_ITER: usize = 100;
    for i in 0..NB_ITER {
        print!(" - {} ... ", i);
        let cdf = {
            if i < 20 {
                None
            } else {
                let scale = (i - 20) as f32 / (NB_ITER - 20) as f32; 
                let t =  (1.0 - scale).powi(2) * 0.25 + 0.005;
                Some(genetic_drawing_rust::Distribution::from_gradients(&img_mag, t))
            }
        };
        let mut dna = genetic_drawing_rust::DNAContext::new(
            &gen,
            10,
            &mut rng,
            i as f32 / NB_ITER as f32,
            images.last(),
            cdf.as_ref(),
        );
        dna.iterate(20, &mut rng);
        println!("{}", dna.error());
        images.push(dna.to_image());
        images
            .last()
            .unwrap()
            .save(&format!("test_{}.png", i))
            .unwrap();
        if let Some(cdf) = cdf {
            cdf.to_image().save(&format!("dist_{}.png", i)).unwrap();
        }
    }

    const NB_ITER_STAGE2: usize = 40;
    let cdf = image::open("./data/mask.jpg").unwrap().to_luma();
    let cdf = genetic_drawing_rust::Distribution::from_image(&cdf);
    gen.bruch_range = (
        genetic_drawing_rust::MinMax {
            min: 0.1,
            max: 0.2
        },
        genetic_drawing_rust::MinMax {
            min: 0.05,
            max: 0.1
        }
    );
    for i in 0..NB_ITER_STAGE2 {
        print!(" - {} ... ", i);
        let mut dna = genetic_drawing_rust::DNAContext::new(
            &gen,
            10,
            &mut rng,
            i as f32 / NB_ITER_STAGE2 as f32,
            images.last(),
            Some(&cdf),
        );
        dna.iterate(30, &mut rng);
        println!("{}", dna.error());
        images.push(dna.to_image());
        images
            .last()
            .unwrap()
            .save(&format!("test_{}.png", i+NB_ITER))
            .unwrap();
    }
    

    // // Test of the different brushes
    // for i in 0..4 {
    //     let stroke = genetic_drawing_rust::Stroke::new(&mut rng, &gen, 0.5, 1);
    //     let drawing = stroke.draw(&dna.brushes);
    //     let mut black = image::DynamicImage::new_luma_a8(drawing.width(), drawing.height()).into_luma_alpha();
    //     for (pdest, porg) in black.pixels_mut().zip(drawing.pixels()) {
    //         pdest[0] += ((porg[1] as f32 / 255.0) * porg[0] as f32) as u8;
    //         pdest[1] = 255;
    //     }
    //     black.save(&format!("test_{}.png", i+1)).unwrap();
    // }
}
