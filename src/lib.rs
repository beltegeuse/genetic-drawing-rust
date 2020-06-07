use image::{DynamicImage, GenericImage, GrayAlphaImage, GrayImage, Pixel};
use imageproc;
use imageproc::geometric_transformations::*;
use rand::prelude::*;

pub struct FloatImage {
    pub values: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl FloatImage {
    pub fn to_image(&self) -> DynamicImage {
        let mut img = DynamicImage::new_luma8(self.width, self.height);
        for x in 0..self.width {
            for y in 0..self.height {
                let v = self.values[(x * self.height + y) as usize];
                let v = (v.min(1.0).powf(1.0 / 2.2) * 255.0) as u8;
                img.put_pixel(x, y, Pixel::from_channels(v, v, v, 255));
            }
        }
        img
    }
}

pub struct GradientImage {
    pub mag: FloatImage,
    pub angle: FloatImage,
}

impl GradientImage {
    pub fn new(img: &image::GrayImage) -> Self {
        let w = img.width() as u32;
        let h = img.height() as u32;

        let gx = imageproc::gradients::horizontal_sobel(&img);
        let gy = imageproc::gradients::vertical_sobel(&img);

        let mut mag = vec![0.0; (w * h) as usize];
        let mut angle = vec![0.0; (w * h) as usize];
        for x in 0..img.width() {
            for y in 0..img.height() {
                let xv = gx.get_pixel(x, y)[0] as f32;
                let yv = gy.get_pixel(x, y)[0] as f32;

                mag[(x * h + y) as usize] = (xv * xv + yv * yv).sqrt();
                angle[(x * h + y) as usize] = yv.atan2(xv).to_degrees();
            }
        }
        let mag_max = *mag.iter().max_by(|i, j| i.partial_cmp(j).unwrap()).unwrap();
        let mag = mag
            .into_iter()
            .map(|v| (v / mag_max).powf(0.3))
            .collect::<Vec<_>>();

        GradientImage {
            mag: FloatImage {
                values: mag,
                width: w,
                height: h,
            },
            angle: FloatImage {
                values: angle,
                width: w,
                height: h,
            },
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.mag.width, self.mag.height)
    }
}

pub struct MinMax {
    pub min: f32,
    pub max: f32,
}
impl MinMax {
    pub fn value(&self, t: f32) -> f32 {
        assert!(t >= 0.0 && t <= 1.0);
        (self.max - self.min) * t + self.min
    }
    pub fn mix(begin: &MinMax, end: &MinMax, time: f32) -> Self {
        assert!(time >= 0.0 && time <= 1.0);
        Self {
            min: end.min * time + begin.min * (1.0 - time),
            max: end.max * time + begin.max * (1.0 - time),
        }
    }
}

pub struct GeneticDrawing {
    pub img_grey: GrayImage,
    pub img_gradient: GradientImage,
    pub bruch_range: (MinMax, MinMax),
    pub brushes: Vec<GrayAlphaImage>,
}

impl GeneticDrawing {
    pub fn load(filename: &str) -> Self {
        // Load image
        let img_grey = image::open(filename).unwrap().into_luma();
        let img_gradient = GradientImage::new(&img_grey);
        Self {
            img_grey,
            img_gradient,
            bruch_range: (MinMax { min: 0.3, max: 0.7 }, MinMax { min: 0.1, max: 0.3 }),
            brushes: vec![],
        }
    }

    pub fn register_brush(&mut self, filename: &str) {
        let mut img_grey = image::open(filename).unwrap().into_luma_alpha();
        img_grey.pixels_mut().for_each(|p| {
            if p[0] == 0 {
                p[1] = 0;
            }
        });
        self.brushes.push(img_grey);
    }

    pub fn clear_brushes(&mut self) {
        self.brushes.clear();
    }
}

pub struct DNAContext<'draw> {
    org_image: GrayImage,
    pub image: GrayImage,
    offset: u32,
    strokes: Vec<Stroke>,
    strokes_scales: MinMax,
    error: f32,
    pub gen: &'draw GeneticDrawing,
}

pub fn blend_alpha(dest: &mut GrayImage, org: &GrayAlphaImage, xoff: u32, yoff: u32) {
    // Safe version of image blending
    for x in 0..org.width() {
        for y in 0..org.height() {
            if x + xoff < dest.width() && y + yoff < dest.height() {
                let porg = org.get_pixel(x, y);
                let alpha = (porg[1] as f32 / 255.0);
                let pdest = dest.get_pixel_mut(x + xoff, y + yoff);
                pdest[0] = (alpha * porg[0] as f32 + (1.0 - alpha) * pdest[0] as f32) as u8;
            }
        }
    }
}

impl<'draw> DNAContext<'draw> {
    pub fn new(
        gen: &'draw GeneticDrawing,
        brushcount: usize,
        rng: &mut rand::rngs::ThreadRng,
        time: f32,
        prev_image: Option<&GrayImage>
    ) -> Self {
        // Compute the max size bruches
        let strokes_scales = MinMax::mix(&gen.bruch_range.0, &gen.bruch_range.1, time);
        let max_size_bruches = strokes_scales.value(1.0);
        let offset = gen
            .brushes
            .iter()
            .map(|b| (b.width().max(b.height()) as f32 * max_size_bruches) as u32)
            .max_by(|i, j| i.partial_cmp(j).unwrap())
            .unwrap();

        // Create a bigger image for easier spatting
        let mut image = image::DynamicImage::new_luma8(
            gen.img_grey.width() + 2 * offset,
            gen.img_grey.height() + 2 * offset,
        ).into_luma();
        if let Some(prev_image) = prev_image {
            // If provided, initialize with previous iteration
            for x in 0..prev_image.width() {
                for y in 0..prev_image.height() {
                    image.get_pixel_mut(x+offset, y+offset)[0] = prev_image.get_pixel(x, y)[0];
                }
            }
        }
        let org_image = image.clone();

        // Create initial strokes and fill the image
        let strokes = (0..brushcount)
            .map(|_| {
                Stroke::new(
                    rng,
                    &strokes_scales,
                    (image.width(), image.height()),
                    gen.brushes.len(),
                )
            })
            .collect::<Vec<_>>();
        for s in &strokes {
            blend_alpha(&mut image, &s.draw(&gen.brushes), s.pos.0, s.pos.1);
        }

        let mut dna = DNAContext {
            org_image,
            image,
            offset,
            strokes,
            strokes_scales,
            error: 0.0,
            gen,
        };
        dna.error = dna.compute_error(&dna.image);
        dna
    }

    pub fn iterate(&mut self, number_iter: usize, rng: &mut rand::rngs::ThreadRng) {
        for _ in 0..number_iter {
            for i in 0..self.strokes.len() {
                // Clone the original image
                let mut new_image = self.org_image.clone();
                // Mutate one stroke
                let mut new_strokes = self.strokes.clone();
                new_strokes[i] = self.strokes[i].mutate(
                    rng,
                    &self.strokes_scales,
                    (self.image.width(), self.image.height()),
                    self.gen.brushes.len(),
                );
                // Update the images with all strokes
                for s in &new_strokes {
                    blend_alpha(&mut new_image, &s.draw(&self.gen.brushes), s.pos.0, s.pos.1);
                }
                // Compute the error
                let new_error = self.compute_error(&new_image);
                if self.error > new_error {
                    self.error = new_error;
                    self.strokes = new_strokes;
                }
            }
        }
    }

    pub fn error(&self) -> f32 {
        self.error
    }

    pub fn to_image(self) -> GrayImage {
        let mut image =
            image::DynamicImage::new_luma8(self.gen.img_grey.width(), self.gen.img_grey.height())
                .into_luma();
        for x in 0..image.width() {
            for y in 0..image.height() {
                image.get_pixel_mut(x, y)[0] =
                    self.image.get_pixel(x + self.offset, y + self.offset)[0];
            }
        }
        image
    }

    fn compute_error(&self, image: &GrayImage) -> f32 {
        assert_eq!(image.width(), self.gen.img_grey.width() + 2 * self.offset);
        assert_eq!(image.height(), self.gen.img_grey.height() + 2 * self.offset);
        // The image is bigger (offset)
        let mut diff: f32 = 0.0;
        for x in 0..self.gen.img_grey.width() {
            for y in 0..self.gen.img_grey.height() {
                let p_diff = self.gen.img_grey.get_pixel(x, y)[0] as f32
                    - image.get_pixel(x + self.offset, y + self.offset)[0] as f32;
                diff += p_diff.abs();
            }
        }
        diff
    }
}

// This object represent painting strokes
#[derive(Clone, Debug)]
pub struct Stroke {
    pub value: f32,
    pub size: f32,
    pub pos: (u32, u32),
    pub rotation: f32,
    pub brush_id: usize,
}

impl Stroke {
    pub fn new(
        rng: &mut rand::rngs::ThreadRng,
        scale: &MinMax,
        img_size: (u32, u32),
        nb_bruches: usize,
    ) -> Stroke {
        let value = rng.gen_range(0.0, 1.0);
        let size = scale.value(rng.gen_range(0.0, 1.0));
        // TODO: use mask
        let pos = (rng.gen_range(0, img_size.0), rng.gen_range(0, img_size.1));
        // TODO: random rotation for now
        let rotation = rng.gen_range(0.0, 360.0);
        let brush_id = rng.gen_range(0, nb_bruches);
        Stroke {
            value,
            size,
            pos,
            rotation,
            brush_id,
        }
    }

    pub fn mutate(
        &self,
        rng: &mut rand::rngs::ThreadRng,
        scale: &MinMax,
        img_size: (u32, u32),
        nb_bruches: usize,
    ) -> Stroke {
        let mut mutations = vec![0, 1, 2, 3, 4, 5];
        mutations.shuffle(rng);
        let nb_mutations = rng.gen_range(0, mutations.len());

        let mut new_stroke = self.clone();
        for i in 0..nb_mutations {
            match mutations[i] {
                0 => new_stroke.value = rng.gen_range(0.0, 1.0),
                1 => new_stroke.size = scale.value(rng.gen_range(0.0, 1.0)),
                2 | 3 => {
                    new_stroke.pos = (rng.gen_range(0, img_size.0), rng.gen_range(0, img_size.1))
                }
                4 => new_stroke.rotation = rng.gen_range(0.0, 360.0),
                5 => new_stroke.brush_id = rng.gen_range(0, nb_bruches),
                _ => panic!("Not covered mutation approach"),
            };
        }

        new_stroke
    }

    pub fn draw(&self, brushes: &Vec<GrayAlphaImage>) -> GrayAlphaImage {
        let b = &brushes[self.brush_id];
        let trans = Projection::translate(b.width() as f32 * 0.5, b.height() as f32 * 0.5)
            * Projection::rotate(self.rotation.to_radians())
            * Projection::scale(self.size, self.size);
        let b = imageproc::geometric_transformations::warp(
            b,
            &trans,
            imageproc::geometric_transformations::Interpolation::Bilinear,
            image::LumaA([0, 0]),
        );
        // Crop the image using bounding box
        let pos = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
        let pos = pos
            .into_iter()
            .map(|(x, y)| trans * (x * b.width() as f32, y * b.height() as f32))
            .collect::<Vec<_>>();
        let xmin = pos
            .iter()
            .map(|p| p.0)
            .min_by(|i, j| i.partial_cmp(j).unwrap())
            .unwrap()
            .max(0.0) as u32;
        let ymin = pos
            .iter()
            .map(|p| p.1)
            .min_by(|i, j| i.partial_cmp(j).unwrap())
            .unwrap()
            .max(0.0) as u32;
        let xmax = pos
            .iter()
            .map(|p| p.0)
            .max_by(|i, j| i.partial_cmp(j).unwrap())
            .unwrap()
            .min(b.height() as f32) as u32;
        let ymax = pos
            .iter()
            .map(|p| p.1)
            .max_by(|i, j| i.partial_cmp(j).unwrap())
            .unwrap()
            .min(b.width() as f32) as u32;
        let mut b = image::imageops::crop_imm(&b, xmin, ymin, xmax - xmin, ymax - ymin).to_image();
        // Use the color and use alpha of the bruch
        b.pixels_mut().for_each(|p| {
            p[1] = p[0];
            p[0] = (self.value * 255.0) as u8;
        });
        b
    }
}
