extern crate genetic_drawing_rust;
use rand::prelude::*;

fn main() {
    // Statements here are executed when the compiled binary is called
    let mut rng = thread_rng();

    // Print text to the console
    let mut gen = genetic_drawing_rust::GeneticDrawing::load("./data/linear.jpg");
    gen.img_gradient.angle.to_image().save("ang.png").unwrap();
    gen.img_gradient.mag.to_image().save("mag.png").unwrap();

    // Register strokes
    println!("Brushes...");
    for i in 0..4 {
        gen.register_brush(&format!("./brushes/watercolor/{}.jpg",i+1));
    }
    println!("DNA...");
    let mut images = vec![];
    const NB_ITER: usize = 1;
    for i in 0..NB_ITER {
        print!(" - {} ... ", i);
        let mut dna = genetic_drawing_rust::DNAContext::new(&gen, 10, &mut rng, i as f32 / NB_ITER as f32, images.last());
        dna.iterate(0, &mut rng);
        println!("{}", dna.error());
        images.push(dna.to_image());
        images.last().unwrap().save(&format!("test_{}.png", i)).unwrap();   
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
